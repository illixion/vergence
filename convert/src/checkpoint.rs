//! PyTorch checkpoints read without Python, as VergenceCore's
//! `TorchCheckpoint` reads them: a `.pth` that `torch.save` wrote (a zip of
//! stored entries: `archive/data.pkl`, the state dict pickled, and one raw
//! little-endian file per tensor storage). The pickle is read by a small
//! interpreter that knows only what a state dict uses; nothing in it runs.

use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    /// Row-major values, whatever the stored type was.
    pub values: Vec<f32>,
}

/// Every state dict in the file: the top level under "" and each one a
/// wrapper holds ({"params_ema": {...}}), by name.
pub struct Checkpoint {
    pub wrapped: HashMap<String, HashMap<String, Tensor>>,
}

impl Checkpoint {
    pub fn read(bytes: &[u8]) -> Result<Checkpoint, String> {
        let zip = Zip::read(bytes)?;
        let pickle = zip.names().into_iter().find(|n| n.ends_with("/data.pkl") || *n == "data.pkl")
            .ok_or("not a PyTorch checkpoint (no data.pkl in the archive)")?.to_string();
        let prefix = &pickle[..pickle.len() - "data.pkl".len()];
        let root = Unpickler { bytes: zip.entry(&pickle)?, at: 0, stack: Vec::new(), memo: HashMap::new() }.run()?;
        let tensors = |value: &Value| -> Result<Option<HashMap<String, Tensor>>, String> {
            let Value::Dict(pairs) = value else { return Ok(None) };
            let mut out = HashMap::new();
            for (key, item) in pairs {
                if let (Value::Str(name), Value::Tensor(t)) = (key, item) {
                    out.insert(name.clone(), t.materialize(&zip, prefix)?);
                }
            }
            Ok((!out.is_empty()).then_some(out))
        };
        let mut wrapped = HashMap::new();
        if let Some(direct) = tensors(&root)? {
            wrapped.insert(String::new(), direct);
        }
        if let Value::Dict(pairs) = &root {
            for (key, value) in pairs {
                if let (Value::Str(name), Some(inner)) = (key, tensors(value)?) {
                    wrapped.insert(name.clone(), inner);
                }
            }
        }
        if wrapped.is_empty() {
            return Err("the checkpoint holds no tensors".into());
        }
        Ok(Checkpoint { wrapped })
    }

    /// The state dict under the first of the usual wrappers the file has.
    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        for key in ["params_ema", "params", "state_dict", ""] {
            if let Some(found) = self.wrapped.get(key) {
                return found;
            }
        }
        unreachable!("read refuses a checkpoint without tensors")
    }
}

struct Zip<'a> {
    bytes: &'a [u8],
    entries: HashMap<String, (usize, usize, u16)>,
}

impl<'a> Zip<'a> {
    fn read(bytes: &'a [u8]) -> Result<Zip<'a>, String> {
        let u16_at = |at: usize| -> usize { bytes.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize).unwrap_or(0) };
        let u32_at = |at: usize| -> usize { u16_at(at) | u16_at(at + 2) << 16 };
        let u64_at = |at: usize| -> usize { u32_at(at) | u32_at(at + 4) << 32 };
        let start = bytes.len().saturating_sub(65_557);
        let eocd = (start..bytes.len().saturating_sub(21)).rev()
            .find(|&i| bytes[i..].starts_with(b"PK\x05\x06"))
            .ok_or("not a zip archive")?;
        let mut count = u16_at(eocd + 10);
        let mut directory = u32_at(eocd + 16);
        if (directory == 0xFFFF_FFFF || count == 0xFFFF) && eocd >= 20 && bytes[eocd - 20..].starts_with(b"PK\x06\x07") {
            let record = u64_at(eocd - 20 + 8);
            count = u64_at(record + 32);
            directory = u64_at(record + 48);
        }
        let mut entries = HashMap::new();
        let mut at = directory;
        for _ in 0..count {
            if at + 46 > bytes.len() || u32_at(at) != 0x0201_4B50 {
                return Err("a broken zip directory".into());
            }
            let method = u16_at(at + 10) as u16;
            let mut size = u32_at(at + 20);
            let (name_len, extra_len, comment_len) = (u16_at(at + 28), u16_at(at + 30), u16_at(at + 32));
            let mut local = u32_at(at + 42);
            let name = String::from_utf8_lossy(&bytes[at + 46..at + 46 + name_len]).into_owned();
            let mut extra = at + 46 + name_len;
            let extra_end = extra + extra_len;
            while extra + 4 <= extra_end {
                let (id, len) = (u16_at(extra), u16_at(extra + 2));
                if id == 1 {
                    let mut field = extra + 4;
                    if u32_at(at + 24) == 0xFFFF_FFFF {
                        field += 8;
                    }
                    if size == 0xFFFF_FFFF {
                        size = u64_at(field);
                        field += 8;
                    }
                    if local == 0xFFFF_FFFF {
                        local = u64_at(field);
                    }
                }
                extra += 4 + len;
            }
            if local + 30 > bytes.len() {
                return Err("a broken zip entry".into());
            }
            let data = local + 30 + u16_at(local + 26) + u16_at(local + 28);
            entries.insert(name, (data, size, method));
            at = extra_end + comment_len;
        }
        Ok(Zip { bytes, entries })
    }

    fn names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    fn entry(&self, name: &str) -> Result<&'a [u8], String> {
        let &(offset, size, method) = self.entries.get(name).ok_or_else(|| format!("no {name} in the archive"))?;
        if method != 0 {
            return Err(format!("{name} is compressed; torch.save stores entries"));
        }
        self.bytes.get(offset..offset + size).ok_or_else(|| format!("{name} runs past the end"))
    }
}

#[derive(Clone, Debug)]
struct TensorRef {
    key: String,
    dtype: String,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl TensorRef {
    fn materialize(&self, zip: &Zip, prefix: &str) -> Result<Tensor, String> {
        let raw = zip.entry(&format!("{prefix}data/{}", self.key))?;
        let size = match self.dtype.as_str() {
            "FloatStorage" => 4,
            "HalfStorage" | "BFloat16Storage" => 2,
            "DoubleStorage" => 8,
            other => return Err(format!("tensors of {other} aren't read")),
        };
        let count: usize = self.shape.iter().product();
        let mut values = Vec::with_capacity(count);
        let mut index = vec![0usize; self.shape.len()];
        for _ in 0..count {
            let element = self.offset + index.iter().zip(&self.stride).map(|(i, s)| i * s).sum::<usize>();
            let b = raw.get(element * size..element * size + size).ok_or("a tensor runs past its storage")?;
            values.push(match size {
                4 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                8 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
                _ if self.dtype == "HalfStorage" => half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
                _ => half::bf16::from_le_bytes([b[0], b[1]]).to_f32(),
            });
            for d in (0..self.shape.len()).rev() {
                index[d] += 1;
                if index[d] < self.shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
        Ok(Tensor { shape: self.shape.clone(), values })
    }
}

#[derive(Clone, Debug)]
enum Value {
    None,
    Bool,
    Int(i64),
    Float,
    Str(String),
    Bytes,
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage(String, String),
    Tensor(TensorRef),
    Object,
    Mark,
}

struct Unpickler<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<Value>,
    memo: HashMap<usize, Value>,
}

impl Unpickler<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let slice = self.bytes.get(self.at..self.at + n).ok_or("the pickle ends early")?;
        self.at += n;
        Ok(slice)
    }

    fn uint(&mut self, n: usize) -> Result<usize, String> {
        Ok(self.take(n)?.iter().rev().fold(0usize, |v, b| v << 8 | *b as usize))
    }

    fn line(&mut self) -> Result<String, String> {
        let end = self.bytes[self.at..].iter().position(|b| *b == b'\n').ok_or("the pickle ends early")?;
        let text = String::from_utf8_lossy(&self.bytes[self.at..self.at + end]).into_owned();
        self.at += end + 1;
        Ok(text)
    }

    fn pop(&mut self) -> Result<Value, String> {
        self.stack.pop().ok_or_else(|| "the pickle's stack runs out".to_string())
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>, String> {
        let i = self.stack.iter().rposition(|v| matches!(v, Value::Mark)).ok_or("the pickle has no mark")?;
        let items = self.stack.split_off(i + 1);
        self.stack.pop();
        Ok(items)
    }

    fn set_items(target: Value, items: Vec<Value>) -> Value {
        let Value::Dict(mut pairs) = target else { return target };
        let mut it = items.into_iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            pairs.push((k, v));
        }
        Value::Dict(pairs)
    }

    fn reduce(callable: Value, arguments: Value) -> Value {
        let (Value::Global(module, name), Value::Tuple(args)) = (callable, arguments) else { return Value::Object };
        match (module.as_str(), name.as_str()) {
            ("collections", "OrderedDict") | ("builtins", "dict") | ("__builtin__", "dict") => Value::Dict(Vec::new()),
            ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
                let ints = |v: &Value| -> Option<Vec<usize>> {
                    let Value::Tuple(items) = v else { return None };
                    items.iter().map(|i| if let Value::Int(n) = i { usize::try_from(*n).ok() } else { None }).collect()
                };
                match (args.first(), args.get(1), args.get(2).and_then(ints), args.get(3).and_then(ints)) {
                    (Some(Value::Storage(key, dtype)), Some(Value::Int(offset)), Some(shape), Some(stride))
                        if shape.len() == stride.len() && *offset >= 0 =>
                    {
                        Value::Tensor(TensorRef { key: key.clone(), dtype: dtype.clone(), offset: *offset as usize, shape, stride })
                    }
                    _ => Value::Object,
                }
            }
            _ => Value::Object,
        }
    }

    fn run(mut self) -> Result<Value, String> {
        loop {
            let op = *self.take(1)?.first().unwrap();
            match op {
                0x80 => {
                    self.take(1)?;
                }
                0x95 => {
                    self.take(8)?;
                }
                0x2E => return self.pop(),
                0x28 => self.stack.push(Value::Mark),
                0x7D => self.stack.push(Value::Dict(Vec::new())),
                0x29 => self.stack.push(Value::Tuple(Vec::new())),
                0x5D => self.stack.push(Value::List(Vec::new())),
                0x4E => self.stack.push(Value::None),
                0x88 | 0x89 => self.stack.push(Value::Bool),
                0x4B => {
                    let v = self.uint(1)?;
                    self.stack.push(Value::Int(v as i64));
                }
                0x4D => {
                    let v = self.uint(2)?;
                    self.stack.push(Value::Int(v as i64));
                }
                0x4A => {
                    let v = self.uint(4)? as u32 as i32;
                    self.stack.push(Value::Int(v as i64));
                }
                0x8A => {
                    let n = self.uint(1)?;
                    let raw = self.take(n)?.to_vec();
                    let mut v: i64 = 0;
                    for (i, b) in raw.iter().enumerate().take(8) {
                        v |= (*b as i64) << (8 * i);
                    }
                    if n > 0 && n < 8 && raw[n - 1] & 0x80 != 0 {
                        v -= 1 << (8 * n);
                    }
                    self.stack.push(Value::Int(v));
                }
                0x47 => {
                    self.take(8)?;
                    self.stack.push(Value::Float);
                }
                0x58 => {
                    let n = self.uint(4)?;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                0x8C => {
                    let n = self.uint(1)?;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                0x8D => {
                    let n = self.uint(8)?;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                0x43 => {
                    let n = self.uint(1)?;
                    self.take(n)?;
                    self.stack.push(Value::Bytes);
                }
                0x42 => {
                    let n = self.uint(4)?;
                    self.take(n)?;
                    self.stack.push(Value::Bytes);
                }
                0x71 => {
                    let k = self.uint(1)?;
                    if let Some(top) = self.stack.last().cloned() {
                        self.memo.insert(k, top);
                    }
                }
                0x72 => {
                    let k = self.uint(4)?;
                    if let Some(top) = self.stack.last().cloned() {
                        self.memo.insert(k, top);
                    }
                }
                0x94 => {
                    let k = self.memo.len();
                    if let Some(top) = self.stack.last().cloned() {
                        self.memo.insert(k, top);
                    }
                }
                0x68 => {
                    let k = self.uint(1)?;
                    self.stack.push(self.memo.get(&k).cloned().unwrap_or(Value::None));
                }
                0x6A => {
                    let k = self.uint(4)?;
                    self.stack.push(self.memo.get(&k).cloned().unwrap_or(Value::None));
                }
                0x63 => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global(module, name));
                }
                0x93 => {
                    let (name, module) = (self.pop()?, self.pop()?);
                    let (Value::Str(m), Value::Str(n)) = (module, name) else { return Err("a bad global".into()) };
                    self.stack.push(Value::Global(m, n));
                }
                0x74 => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                0x86 => {
                    let (b, a) = (self.pop()?, self.pop()?);
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let (c, b, a) = (self.pop()?, self.pop()?, self.pop()?);
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                0x6C => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(items));
                }
                0x61 => {
                    let item = self.pop()?;
                    let list = self.pop()?;
                    self.stack.push(match list {
                        Value::List(mut items) => {
                            items.push(item);
                            Value::List(items)
                        }
                        _ => Value::Object,
                    });
                }
                0x65 => {
                    let items = self.pop_mark()?;
                    let list = self.pop()?;
                    self.stack.push(match list {
                        Value::List(mut all) => {
                            all.extend(items);
                            Value::List(all)
                        }
                        _ => Value::Object,
                    });
                }
                0x73 => {
                    let (value, key) = (self.pop()?, self.pop()?);
                    let target = self.pop()?;
                    self.stack.push(Self::set_items(target, vec![key, value]));
                }
                0x75 => {
                    let items = self.pop_mark()?;
                    let target = self.pop()?;
                    self.stack.push(Self::set_items(target, items));
                }
                0x52 => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    self.stack.push(Self::reduce(callable, args));
                }
                0x81 => {
                    self.pop()?;
                    self.pop()?;
                    self.stack.push(Value::Object);
                }
                0x62 => {
                    self.pop()?;
                }
                0x51 => {
                    let Value::Tuple(parts) = self.pop()? else { return Err("a storage reference it can't read".into()) };
                    match (parts.get(1), parts.get(2)) {
                        (Some(Value::Global(_, dtype)), Some(Value::Str(key))) => self.stack.push(Value::Storage(key.clone(), dtype.clone())),
                        _ => return Err("a storage reference it can't read".into()),
                    }
                }
                0x30 => {
                    self.pop()?;
                }
                0x31 => {
                    self.pop_mark()?;
                }
                0x32 => {
                    let top = self.stack.last().cloned().unwrap_or(Value::None);
                    self.stack.push(top);
                }
                other => return Err(format!("pickle opcode 0x{other:02X} isn't read")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint VergenceCore's tests read (torch 2.7): params_ema with a
    /// float32 [2,3], a float16 [4], a transposed view and its base.
    fn fixture() -> Vec<u8> {
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../Tests/VergenceAppleTests/Fixtures/tiny-srvgg.pth")).unwrap()
    }

    #[test]
    fn reads_the_tiny_network() {
        let checkpoint = Checkpoint::read(&fixture()).unwrap();
        let t = checkpoint.tensors();
        assert_eq!(t["body.0.weight"].shape, vec![4, 3, 3, 3]);
        assert_eq!(t["body.1.weight"].shape, vec![4]);
        assert_eq!(t["body.4.weight"].shape, vec![48, 4, 3, 3]);
        assert_eq!(t["body.4.bias"].values.len(), 48);
    }

    #[test]
    fn refuses_what_isnt_one() {
        assert!(Checkpoint::read(b"not a zip").is_err());
        assert!(Checkpoint::read(&fixture()[..600]).is_err());
    }
}
