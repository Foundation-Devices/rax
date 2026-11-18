//! An x86-64 assembler with only the instructions that the compiler uses.

pub type Reg = u8;
pub const RAX: Reg = 0;
pub const RCX: Reg = 1;
pub const RDX: Reg = 2;
pub const RBX: Reg = 3;
pub const RSP: Reg = 4;
pub const RBP: Reg = 5;
pub const RSI: Reg = 6;
pub const RDI: Reg = 7;
pub const R8: Reg = 8;
pub const R9: Reg = 9;
pub const R10: Reg = 10;
pub const R11: Reg = 11;
pub const R12: Reg = 12;
pub const R13: Reg = 13;
pub const R14: Reg = 14;
pub const R15: Reg = 15;

/// An operand in a register or in the memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rm {
    Reg(Reg),
    /// The address is `base + index + disp`.
    Mem {
        base: Reg,
        index: Option<Reg>,
        disp: i32,
    },
}

pub fn mem(base: Reg, disp: usize) -> Rm {
    Rm::Mem {
        base,
        index: None,
        disp: disp as i32,
    }
}

pub fn indexed(base: Reg, index: Reg, disp: usize) -> Rm {
    Rm::Mem {
        base,
        index: Some(index),
        disp: disp as i32,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    Byte,
    Word,
    Long,
    Quad,
}
pub use Size::{Byte, Long, Quad, Word};

/// The operations with two operands, by their number in the opcodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alu {
    Add = 0,
    Or = 1,
    Adc = 2,
    Sbb = 3,
    And = 4,
    Sub = 5,
    Xor = 6,
    Cmp = 7,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shift {
    Ror = 1,
    Rcr = 3,
    Shl = 4,
    Shr = 5,
    Sar = 7,
}

/// The operations with one operand of the opcode 0xf7.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    Not = 2,
    Neg = 3,
    Div = 6,
    Idiv = 7,
}

// Conditions.
pub const O: u8 = 0;
pub const NO: u8 = 1;
pub const B: u8 = 2;
pub const AE: u8 = 3;
pub const E: u8 = 4;
pub const NE: u8 = 5;
pub const BE: u8 = 6;
pub const A: u8 = 7;
pub const S: u8 = 8;
pub const NS: u8 = 9;
pub const L: u8 = 12;
pub const GE: u8 = 13;
pub const LE: u8 = 14;
pub const G: u8 = 15;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Label(usize);

pub struct Asm {
    /// The address of the first byte.
    base: usize,
    pub bytes: Vec<u8>,
    labels: Vec<Option<usize>>,
    /// The branches to labels: the offset of the 32-bit field and the label.
    fixups: Vec<(usize, usize)>,
    /// Set by each instruction that can change the flags of the host, and by `bind`:
    /// a branch can come there with other flags.
    pub flags_changed: bool,
}

impl Asm {
    pub fn new(base: usize) -> Self {
        Asm {
            base,
            bytes: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
            flags_changed: true,
        }
    }

    pub fn base(&self) -> usize {
        self.base
    }

    /// The address of the next byte.
    pub fn here(&self) -> usize {
        self.base + self.bytes.len()
    }

    pub fn emit(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    pub fn emit_u32(&mut self, value: u32) {
        self.emit(&value.to_le_bytes());
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label.0] = Some(self.bytes.len());
        self.flags_changed = true;
    }

    /// Writes the offsets of the branches to labels. Each label must have a place.
    pub fn finish(&mut self) {
        for &(at, label) in &self.fixups {
            let offset = self.labels[label].unwrap() as i64 - (at as i64 + 4);
            self.bytes[at..at + 4].copy_from_slice(&(offset as i32).to_le_bytes());
        }
        self.fixups.clear();
    }

    /// The prefixes, the opcode, and the bytes that give the register `reg` and the
    /// operand `rm`.
    fn modrm(&mut self, size: Size, opcode: &[u8], reg: Reg, rm: Rm) {
        self.encode(size, opcode, reg, rm, size == Byte, size == Byte);
    }

    /// The same for an opcode with one operand: `extension` is a part of the opcode.
    fn modrm_one(&mut self, size: Size, opcode: &[u8], extension: u8, rm: Rm) {
        self.encode(size, opcode, extension, rm, false, size == Byte);
    }

    /// `reg_is_byte`, `rm_is_byte`: the operand, when it is a register, has 8 bits.
    fn encode(
        &mut self,
        size: Size,
        opcode: &[u8],
        reg: u8,
        rm: Rm,
        reg_is_byte: bool,
        rm_is_byte: bool,
    ) {
        if size == Word {
            self.bytes.push(0x66);
        }
        let (index, base) = match rm {
            Rm::Reg(register) => (0, register),
            Rm::Mem { base, index, .. } => (index.unwrap_or(0), base),
        };
        let rex = u8::from(size == Quad) << 3 | (reg >> 3) << 2 | (index >> 3) << 1 | base >> 3;
        // The registers 4 to 7 of 8 bits are AH to BH with no REX prefix.
        let byte_register =
            (reg_is_byte && (4..8).contains(&reg)) || (rm_is_byte && matches!(rm, Rm::Reg(4..8)));
        if rex != 0 || byte_register {
            self.bytes.push(0x40 | rex);
        }
        self.emit(opcode);
        match rm {
            Rm::Reg(register) => self.bytes.push(0xc0 | (reg & 7) << 3 | register & 7),
            Rm::Mem { base, index, disp } => {
                let mode = match disp {
                    0 if base & 7 != RBP => 0,
                    -128..=127 => 1,
                    _ => 2,
                };
                if index.is_some() || base & 7 == RSP {
                    self.bytes.push(mode << 6 | (reg & 7) << 3 | 4);
                    self.bytes.push((index.unwrap_or(RSP) & 7) << 3 | base & 7);
                } else {
                    self.bytes.push(mode << 6 | (reg & 7) << 3 | base & 7);
                }
                match mode {
                    0 => {}
                    1 => self.bytes.push(disp as u8),
                    _ => self.emit_u32(disp as u32),
                }
            }
        }
    }

    /// `dst = dst op src`, or only the flags for CMP.
    pub fn alu(&mut self, op: Alu, size: Size, dst: Reg, src: Rm) {
        self.flags_changed = true;
        let opcode = (op as u8) << 3 | u8::from(size != Byte);
        match src {
            Rm::Reg(src) => self.modrm(size, &[opcode], src, Rm::Reg(dst)),
            Rm::Mem { .. } => self.modrm(size, &[opcode | 2], dst, src),
        }
    }

    /// The same with the first operand in the memory.
    pub fn alu_to(&mut self, op: Alu, size: Size, dst: Rm, src: Reg) {
        self.flags_changed = true;
        self.modrm(size, &[(op as u8) << 3 | u8::from(size != Byte)], src, dst);
    }

    pub fn alu_imm(&mut self, op: Alu, size: Size, dst: Rm, imm: u32) {
        self.flags_changed = true;
        if size == Byte {
            self.modrm_one(size, &[0x80], op as u8, dst);
            self.bytes.push(imm as u8);
        } else if imm as i32 as i8 as i32 == imm as i32 {
            self.modrm_one(size, &[0x83], op as u8, dst);
            self.bytes.push(imm as u8);
        } else {
            self.modrm_one(size, &[0x81], op as u8, dst);
            self.emit_u32(imm);
        }
    }

    /// A load, or a copy of a register. A load of 32 bits clears the high half.
    pub fn mov(&mut self, size: Size, dst: Reg, src: Rm) {
        let opcode = 0x88 | u8::from(size != Byte);
        match src {
            Rm::Reg(src) => self.modrm(size, &[opcode], src, Rm::Reg(dst)),
            Rm::Mem { .. } => self.modrm(size, &[opcode | 2], dst, src),
        }
    }

    pub fn mov_to(&mut self, size: Size, dst: Rm, src: Reg) {
        self.modrm(size, &[0x88 | u8::from(size != Byte)], src, dst);
    }

    /// The high half of the register gets 0. The flags stay.
    pub fn mov_imm(&mut self, dst: Reg, imm: u32) {
        if dst >= 8 {
            self.bytes.push(0x41);
        }
        self.bytes.push(0xb8 | dst & 7);
        self.emit_u32(imm);
    }

    pub fn mov_imm64(&mut self, dst: Reg, imm: u64) {
        if imm >> 32 == 0 {
            return self.mov_imm(dst, imm as u32);
        }
        self.emit(&[0x48 | dst >> 3, 0xb8 | dst & 7]);
        self.emit(&imm.to_le_bytes());
    }

    /// With `Quad`, the value is the 32 bits with their sign.
    pub fn mov_imm_to(&mut self, size: Size, dst: Rm, imm: u32) {
        self.modrm_one(size, &[0xc7], 0, dst);
        self.emit_u32(imm);
    }

    /// A load of 8 or 16 bits (`from`) to a register of 32 bits.
    pub fn extend(&mut self, signed: bool, from: Size, dst: Reg, src: Rm) {
        let opcode = 0xb6 | u8::from(signed) << 3 | u8::from(from == Word);
        self.encode(Long, &[0x0f, opcode], dst, src, false, from == Byte);
    }

    /// A load of 32 bits with their sign to a register of 64 bits.
    pub fn movsxd(&mut self, dst: Reg, src: Rm) {
        self.modrm(Quad, &[0x63], dst, src);
    }

    pub fn lea(&mut self, size: Size, dst: Reg, address: Rm) {
        self.modrm(size, &[0x8d], dst, address);
    }

    pub fn test(&mut self, size: Size, first: Rm, second: Reg) {
        self.flags_changed = true;
        self.modrm(size, &[0x84 | u8::from(size != Byte)], second, first);
    }

    pub fn test_imm(&mut self, first: Rm, imm: u32) {
        self.flags_changed = true;
        self.modrm_one(Long, &[0xf7], 0, first);
        self.emit_u32(imm);
    }

    /// A shift by an amount that is not 0.
    pub fn shift(&mut self, op: Shift, size: Size, dst: Rm, amount: u32) {
        self.flags_changed = true;
        if amount == 1 {
            self.modrm_one(size, &[0xd1], op as u8, dst);
        } else {
            self.modrm_one(size, &[0xc1], op as u8, dst);
            self.bytes.push(amount as u8);
        }
    }

    /// A shift by CL.
    pub fn shift_cl(&mut self, op: Shift, dst: Rm) {
        self.flags_changed = true;
        self.modrm_one(Long, &[0xd3], op as u8, dst);
    }

    /// DIV and IDIV divide EDX:EAX, with the quotient in EAX.
    pub fn unary(&mut self, op: Unary, dst: Rm) {
        self.flags_changed = true;
        self.modrm_one(Long, &[0xf7], op as u8, dst);
    }

    pub fn imul(&mut self, size: Size, dst: Reg, src: Rm) {
        self.flags_changed = true;
        self.modrm(size, &[0x0f, 0xaf], dst, src);
    }

    /// The number of the highest bit that is 1. ZF is set, and `dst` has no value, when
    /// the operand is 0.
    pub fn bsr(&mut self, dst: Reg, src: Rm) {
        self.flags_changed = true;
        self.modrm(Long, &[0x0f, 0xbd], dst, src);
    }

    pub fn bswap(&mut self, dst: Reg) {
        if dst >= 8 {
            self.bytes.push(0x41);
        }
        self.emit(&[0x0f, 0xc8 | dst & 7]);
    }

    pub fn cmov(&mut self, cond: u8, dst: Reg, src: Rm) {
        self.modrm(Long, &[0x0f, 0x40 | cond], dst, src);
    }

    /// Writes 1 or 0 to an operand of 8 bits.
    pub fn setcc(&mut self, cond: u8, dst: Rm) {
        self.modrm_one(Byte, &[0x0f, 0x90 | cond], 0, dst);
    }

    /// CF gets the bit `bit` of the operand. ZF stays.
    pub fn bt(&mut self, operand: Rm, bit: u32) {
        self.flags_changed = true;
        self.modrm_one(Long, &[0x0f, 0xba], 4, operand);
        self.bytes.push(bit as u8);
    }

    /// EDX gets the sign of EAX.
    pub fn cdq(&mut self) {
        self.bytes.push(0x99);
    }

    /// AH gets SF, ZF and CF in its bits 7, 6 and 0.
    pub fn lahf(&mut self) {
        self.bytes.push(0x9f);
    }

    /// SF, ZF and CF get the bits 7, 6 and 0 of AH. OF stays.
    pub fn sahf(&mut self) {
        self.flags_changed = true;
        self.bytes.push(0x9e);
    }

    pub fn cmc(&mut self) {
        self.flags_changed = true;
        self.bytes.push(0xf5);
    }

    pub fn stc(&mut self) {
        self.flags_changed = true;
        self.bytes.push(0xf9);
    }

    /// `cmp al, imm`
    pub fn cmp_al(&mut self, imm: u8) {
        self.flags_changed = true;
        self.emit(&[0x3c, imm]);
    }

    /// `or ah, cl`
    pub fn or_ah_cl(&mut self) {
        self.flags_changed = true;
        self.emit(&[0x08, 0xcc]);
    }

    pub fn push(&mut self, register: Reg) {
        if register >= 8 {
            self.bytes.push(0x41);
        }
        self.bytes.push(0x50 | register & 7);
    }

    pub fn pop(&mut self, register: Reg) {
        if register >= 8 {
            self.bytes.push(0x41);
        }
        self.bytes.push(0x58 | register & 7);
    }

    pub fn ret(&mut self) {
        self.bytes.push(0xc3);
    }

    fn branch_to(&mut self, opcode: &[u8], label: Label) {
        self.emit(opcode);
        self.fixups.push((self.bytes.len(), label.0));
        self.emit_u32(0);
    }

    fn branch_to_address(&mut self, opcode: u8, address: usize) {
        let offset = address as i64 - (self.here() as i64 + 5);
        assert!(i32::try_from(offset).is_ok());
        self.bytes.push(opcode);
        self.emit_u32(offset as u32);
    }

    pub fn jmp(&mut self, label: Label) {
        self.branch_to(&[0xe9], label);
    }

    pub fn jcc(&mut self, cond: u8, label: Label) {
        self.branch_to(&[0x0f, 0x80 | cond], label);
    }

    pub fn jmp_to(&mut self, address: usize) {
        self.branch_to_address(0xe9, address);
    }

    pub fn call_to(&mut self, address: usize) {
        self.flags_changed = true;
        self.branch_to_address(0xe8, address);
    }

    pub fn jmp_rm(&mut self, target: Rm) {
        self.modrm_one(Long, &[0xff], 4, target);
    }

    pub fn call_rm(&mut self, target: Rm) {
        self.flags_changed = true;
        self.modrm_one(Long, &[0xff], 2, target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(emit: impl FnOnce(&mut Asm)) -> Vec<u8> {
        let mut asm = Asm::new(0x1000);
        emit(&mut asm);
        asm.finish();
        asm.bytes
    }

    #[test]
    fn instructions_have_the_bytes_of_the_host_assembler() {
        // The text (Intel syntax), the bytes that clang 17 gives for it with
        // --target=x86_64-unknown-linux-gnu, and the bytes of the assembler.
        #[rustfmt::skip]
        let table: &[(&str, &[u8], Vec<u8>)] = &[
            ("add esi, edi", &[0x01, 0xfe], bytes(|a| a.alu(Alu::Add, Long, RSI, Rm::Reg(RDI)))),
            ("or r8d, ebx", &[0x41, 0x09, 0xd8], bytes(|a| a.alu(Alu::Or, Long, R8, Rm::Reg(RBX)))),
            ("adc ebx, r9d", &[0x44, 0x11, 0xcb], bytes(|a| a.alu(Alu::Adc, Long, RBX, Rm::Reg(R9)))),
            ("sbb r12d, r13d", &[0x45, 0x19, 0xec], bytes(|a| a.alu(Alu::Sbb, Long, R12, Rm::Reg(R13)))),
            ("and ebx, dword ptr [rbp + 44]", &[0x23, 0x5d, 0x2c], bytes(|a| a.alu(Alu::And, Long, RBX, mem(RBP, 44)))),
            ("sub r15d, dword ptr [rbp + 300]", &[0x44, 0x2b, 0xbd, 0x2c, 0x01, 0x00, 0x00], bytes(|a| a.alu(Alu::Sub, Long, R15, mem(RBP, 300)))),
            ("xor edx, dword ptr [rbp]", &[0x33, 0x55, 0x00], bytes(|a| a.alu(Alu::Xor, Long, RDX, mem(RBP, 0)))),
            ("cmp ecx, dword ptr [rdx + 4]", &[0x3b, 0x4a, 0x04], bytes(|a| a.alu(Alu::Cmp, Long, RCX, mem(RDX, 4)))),
            ("add rdx, qword ptr [rsp + 8]", &[0x48, 0x03, 0x54, 0x24, 0x08], bytes(|a| a.alu(Alu::Add, Quad, RDX, mem(RSP, 8)))),
            ("add rdx, r15", &[0x4c, 0x01, 0xfa], bytes(|a| a.alu(Alu::Add, Quad, RDX, Rm::Reg(R15)))),
            ("or rdx, rcx", &[0x48, 0x09, 0xca], bytes(|a| a.alu(Alu::Or, Quad, RDX, Rm::Reg(RCX)))),
            ("cmp ebx, dword ptr [rdx]", &[0x3b, 0x1a], bytes(|a| a.alu(Alu::Cmp, Long, RBX, mem(RDX, 0)))),
            ("add dword ptr [rbp + 64], ebx", &[0x01, 0x5d, 0x40], bytes(|a| a.alu_to(Alu::Add, Long, mem(RBP, 64), RBX))),
            ("or dword ptr [rbp + 64], r10d", &[0x44, 0x09, 0x55, 0x40], bytes(|a| a.alu_to(Alu::Or, Long, mem(RBP, 64), R10))),
            ("add esi, 4", &[0x83, 0xc6, 0x04], bytes(|a| a.alu_imm(Alu::Add, Long, Rm::Reg(RSI), 4))),
            ("sub r15d, -8", &[0x41, 0x83, 0xef, 0xf8], bytes(|a| a.alu_imm(Alu::Sub, Long, Rm::Reg(R15), -8i32 as u32))),
            ("and ecx, 4294963200", &[0x81, 0xe1, 0x00, 0xf0, 0xff, 0xff], bytes(|a| a.alu_imm(Alu::And, Long, Rm::Reg(RCX), 0xffff_f000))),
            ("and ecx, 256", &[0x81, 0xe1, 0x00, 0x01, 0x00, 0x00], bytes(|a| a.alu_imm(Alu::And, Long, Rm::Reg(RCX), 0x100))),
            ("cmp ebx, 305419896", &[0x81, 0xfb, 0x78, 0x56, 0x34, 0x12], bytes(|a| a.alu_imm(Alu::Cmp, Long, Rm::Reg(RBX), 0x1234_5678))),
            ("add dword ptr [rbp + 64], 1000", &[0x81, 0x45, 0x40, 0xe8, 0x03, 0x00, 0x00], bytes(|a| a.alu_imm(Alu::Add, Long, mem(RBP, 64), 1000))),
            ("and dword ptr [rbp + 64], -16", &[0x83, 0x65, 0x40, 0xf0], bytes(|a| a.alu_imm(Alu::And, Long, mem(RBP, 64), !15))),
            ("sub qword ptr [rsp + 32], 64", &[0x48, 0x83, 0x6c, 0x24, 0x20, 0x40], bytes(|a| a.alu_imm(Alu::Sub, Quad, mem(RSP, 32), 64))),
            ("add qword ptr [rsp + 32], 3", &[0x48, 0x83, 0x44, 0x24, 0x20, 0x03], bytes(|a| a.alu_imm(Alu::Add, Quad, mem(RSP, 32), 3))),
            ("add rsp, 56", &[0x48, 0x83, 0xc4, 0x38], bytes(|a| a.alu_imm(Alu::Add, Quad, Rm::Reg(RSP), 56))),
            ("cmp byte ptr [rbp + 200], 16", &[0x80, 0xbd, 0xc8, 0x00, 0x00, 0x00, 0x10], bytes(|a| a.alu_imm(Alu::Cmp, Byte, mem(RBP, 200), 16))),
            ("mov ebx, esi", &[0x89, 0xf3], bytes(|a| a.mov(Long, RBX, Rm::Reg(RSI)))),
            ("mov r8d, r15d", &[0x45, 0x89, 0xf8], bytes(|a| a.mov(Long, R8, Rm::Reg(R15)))),
            ("mov rdx, rsi", &[0x48, 0x89, 0xf2], bytes(|a| a.mov(Quad, RDX, Rm::Reg(RSI)))),
            ("mov ebx, dword ptr [rbp + 44]", &[0x8b, 0x5d, 0x2c], bytes(|a| a.mov(Long, RBX, mem(RBP, 44)))),
            ("mov rbp, qword ptr [rdi]", &[0x48, 0x8b, 0x2f], bytes(|a| a.mov(Quad, RBP, mem(RDI, 0)))),
            ("mov rdx, qword ptr [rdx + 16]", &[0x48, 0x8b, 0x52, 0x10], bytes(|a| a.mov(Quad, RDX, mem(RDX, 16)))),
            ("mov r9d, dword ptr [rdx + rsi]", &[0x44, 0x8b, 0x0c, 0x32], bytes(|a| a.mov(Long, R9, indexed(RDX, RSI, 0)))),
            ("mov ebx, dword ptr [rdx + r15 + 8]", &[0x42, 0x8b, 0x5c, 0x3a, 0x08], bytes(|a| a.mov(Long, RBX, indexed(RDX, R15, 8)))),
            ("mov ecx, dword ptr [r13]", &[0x41, 0x8b, 0x4d, 0x00], bytes(|a| a.mov(Long, RCX, mem(R13, 0)))),
            ("mov ecx, dword ptr [r12]", &[0x41, 0x8b, 0x0c, 0x24], bytes(|a| a.mov(Long, RCX, mem(R12, 0)))),
            ("mov ecx, dword ptr [r12 + 400]", &[0x41, 0x8b, 0x8c, 0x24, 0x90, 0x01, 0x00, 0x00], bytes(|a| a.mov(Long, RCX, mem(R12, 400)))),
            ("mov ecx, dword ptr [r13 + r12]", &[0x43, 0x8b, 0x4c, 0x25, 0x00], bytes(|a| a.mov(Long, RCX, indexed(R13, R12, 0)))),
            ("mov dword ptr [rbp + 60], edx", &[0x89, 0x55, 0x3c], bytes(|a| a.mov_to(Long, mem(RBP, 60), RDX))),
            ("mov qword ptr [rsp + 8], rax", &[0x48, 0x89, 0x44, 0x24, 0x08], bytes(|a| a.mov_to(Quad, mem(RSP, 8), RAX))),
            ("mov dword ptr [rdx + rbx], r14d", &[0x44, 0x89, 0x34, 0x1a], bytes(|a| a.mov_to(Long, indexed(RDX, RBX, 0), R14))),
            ("mov word ptr [rdx + rbx], si", &[0x66, 0x89, 0x34, 0x1a], bytes(|a| a.mov_to(Word, indexed(RDX, RBX, 0), RSI))),
            ("mov word ptr [rdx + r15], r9w", &[0x66, 0x46, 0x89, 0x0c, 0x3a], bytes(|a| a.mov_to(Word, indexed(RDX, R15, 0), R9))),
            ("mov byte ptr [rdx + rbx], sil", &[0x40, 0x88, 0x34, 0x1a], bytes(|a| a.mov_to(Byte, indexed(RDX, RBX, 0), RSI))),
            ("mov byte ptr [rdx + rbx], cl", &[0x88, 0x0c, 0x1a], bytes(|a| a.mov_to(Byte, indexed(RDX, RBX, 0), RCX))),
            ("mov byte ptr [rdx + rbx], r12b", &[0x44, 0x88, 0x24, 0x1a], bytes(|a| a.mov_to(Byte, indexed(RDX, RBX, 0), R12))),
            ("mov byte ptr [rbp + 130], dl", &[0x88, 0x95, 0x82, 0x00, 0x00, 0x00], bytes(|a| a.mov_to(Byte, mem(RBP, 130), RDX))),
            ("mov ebx, 305419896", &[0xbb, 0x78, 0x56, 0x34, 0x12], bytes(|a| a.mov_imm(RBX, 0x1234_5678))),
            ("mov r11d, 0", &[0x41, 0xbb, 0x00, 0x00, 0x00, 0x00], bytes(|a| a.mov_imm(R11, 0))),
            ("movabs rdx, 1311768467463790320", &[0x48, 0xba, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12], bytes(|a| a.mov_imm64(RDX, 0x1234_5678_9abc_def0))),
            ("movabs r10, 1311768467463790320", &[0x49, 0xba, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12], bytes(|a| a.mov_imm64(R10, 0x1234_5678_9abc_def0))),
            ("mov edx, 7", &[0xba, 0x07, 0x00, 0x00, 0x00], bytes(|a| a.mov_imm64(RDX, 7))),
            ("mov dword ptr [rbp + 72], 305419896", &[0xc7, 0x45, 0x48, 0x78, 0x56, 0x34, 0x12], bytes(|a| a.mov_imm_to(Long, mem(RBP, 72), 0x1234_5678))),
            ("mov qword ptr [rdx + 40], 0", &[0x48, 0xc7, 0x42, 0x28, 0x00, 0x00, 0x00, 0x00], bytes(|a| a.mov_imm_to(Quad, mem(RDX, 40), 0))),
            ("movzx ebx, sil", &[0x40, 0x0f, 0xb6, 0xde], bytes(|a| a.extend(false, Byte, RBX, Rm::Reg(RSI)))),
            ("movzx edx, dl", &[0x0f, 0xb6, 0xd2], bytes(|a| a.extend(false, Byte, RDX, Rm::Reg(RDX)))),
            ("movzx r8d, r9b", &[0x45, 0x0f, 0xb6, 0xc1], bytes(|a| a.extend(false, Byte, R8, Rm::Reg(R9)))),
            ("movzx edi, di", &[0x0f, 0xb7, 0xff], bytes(|a| a.extend(false, Word, RDI, Rm::Reg(RDI)))),
            ("movsx ebx, dil", &[0x40, 0x0f, 0xbe, 0xdf], bytes(|a| a.extend(true, Byte, RBX, Rm::Reg(RDI)))),
            ("movsx r12d, bx", &[0x44, 0x0f, 0xbf, 0xe3], bytes(|a| a.extend(true, Word, R12, Rm::Reg(RBX)))),
            ("movzx esi, byte ptr [rdx + rbx]", &[0x0f, 0xb6, 0x34, 0x1a], bytes(|a| a.extend(false, Byte, RSI, indexed(RDX, RBX, 0)))),
            ("movzx r9d, word ptr [rdx + r15]", &[0x46, 0x0f, 0xb7, 0x0c, 0x3a], bytes(|a| a.extend(false, Word, R9, indexed(RDX, R15, 0)))),
            ("movsx ecx, byte ptr [rdx + rbx]", &[0x0f, 0xbe, 0x0c, 0x1a], bytes(|a| a.extend(true, Byte, RCX, indexed(RDX, RBX, 0)))),
            ("movsx ecx, word ptr [rbp + 52]", &[0x0f, 0xbf, 0x4d, 0x34], bytes(|a| a.extend(true, Word, RCX, mem(RBP, 52)))),
            ("movzx eax, byte ptr [rbp + 128]", &[0x0f, 0xb6, 0x85, 0x80, 0x00, 0x00, 0x00], bytes(|a| a.extend(false, Byte, RAX, mem(RBP, 128)))),
            ("movsxd rbx, esi", &[0x48, 0x63, 0xde], bytes(|a| a.movsxd(RBX, Rm::Reg(RSI)))),
            ("movsxd rdx, dword ptr [rbp + 48]", &[0x48, 0x63, 0x55, 0x30], bytes(|a| a.movsxd(RDX, mem(RBP, 48)))),
            ("lea ebx, [rsi + 4]", &[0x8d, 0x5e, 0x04], bytes(|a| a.lea(Long, RBX, mem(RSI, 4)))),
            ("lea ecx, [r15 - 20]", &[0x41, 0x8d, 0x4f, 0xec], bytes(|a| a.lea(Long, RCX, mem(R15, -20i32 as usize)))),
            ("lea ecx, [rbx + 3]", &[0x8d, 0x4b, 0x03], bytes(|a| a.lea(Long, RCX, mem(RBX, 3)))),
            ("lea eax, [rax + rcx]", &[0x8d, 0x04, 0x08], bytes(|a| a.lea(Long, RAX, indexed(RAX, RCX, 0)))),
            ("lea ebx, [r12 + 4096]", &[0x41, 0x8d, 0x9c, 0x24, 0x00, 0x10, 0x00, 0x00], bytes(|a| a.lea(Long, RBX, mem(R12, 4096)))),
            ("lea ebx, [r13]", &[0x41, 0x8d, 0x5d, 0x00], bytes(|a| a.lea(Long, RBX, mem(R13, 0)))),
            ("test esi, esi", &[0x85, 0xf6], bytes(|a| a.test(Long, Rm::Reg(RSI), RSI))),
            ("test r9d, ebx", &[0x41, 0x85, 0xd9], bytes(|a| a.test(Long, Rm::Reg(R9), RBX))),
            ("test dword ptr [rbp + 48], ebx", &[0x85, 0x5d, 0x30], bytes(|a| a.test(Long, mem(RBP, 48), RBX))),
            ("test rdx, rdx", &[0x48, 0x85, 0xd2], bytes(|a| a.test(Quad, Rm::Reg(RDX), RDX))),
            ("test ebx, 3", &[0xf7, 0xc3, 0x03, 0x00, 0x00, 0x00], bytes(|a| a.test_imm(Rm::Reg(RBX), 3))),
            ("test r15d, 3", &[0x41, 0xf7, 0xc7, 0x03, 0x00, 0x00, 0x00], bytes(|a| a.test_imm(Rm::Reg(R15), 3))),
            ("test dword ptr [rbp + 48], 255", &[0xf7, 0x45, 0x30, 0xff, 0x00, 0x00, 0x00], bytes(|a| a.test_imm(mem(RBP, 48), 255))),
            ("shl ebx, 5", &[0xc1, 0xe3, 0x05], bytes(|a| a.shift(Shift::Shl, Long, Rm::Reg(RBX), 5))),
            ("shr edx, 12", &[0xc1, 0xea, 0x0c], bytes(|a| a.shift(Shift::Shr, Long, Rm::Reg(RDX), 12))),
            ("sar r9d, 31", &[0x41, 0xc1, 0xf9, 0x1f], bytes(|a| a.shift(Shift::Sar, Long, Rm::Reg(R9), 31))),
            ("ror edx, 8", &[0xc1, 0xca, 0x08], bytes(|a| a.shift(Shift::Ror, Long, Rm::Reg(RDX), 8))),
            ("rcr edx", &[0xd1, 0xda], bytes(|a| a.shift(Shift::Rcr, Long, Rm::Reg(RDX), 1))),
            ("shl ebx", &[0xd1, 0xe3], bytes(|a| a.shift(Shift::Shl, Long, Rm::Reg(RBX), 1))),
            ("shr rbx, 32", &[0x48, 0xc1, 0xeb, 0x20], bytes(|a| a.shift(Shift::Shr, Quad, Rm::Reg(RBX), 32))),
            ("shl rdx, 32", &[0x48, 0xc1, 0xe2, 0x20], bytes(|a| a.shift(Shift::Shl, Quad, Rm::Reg(RDX), 32))),
            ("shl ebx, cl", &[0xd3, 0xe3], bytes(|a| a.shift_cl(Shift::Shl, Rm::Reg(RBX)))),
            ("shr r8d, cl", &[0x41, 0xd3, 0xe8], bytes(|a| a.shift_cl(Shift::Shr, Rm::Reg(R8)))),
            ("sar esi, cl", &[0xd3, 0xfe], bytes(|a| a.shift_cl(Shift::Sar, Rm::Reg(RSI)))),
            ("ror edi, cl", &[0xd3, 0xcf], bytes(|a| a.shift_cl(Shift::Ror, Rm::Reg(RDI)))),
            ("not ebx", &[0xf7, 0xd3], bytes(|a| a.unary(Unary::Not, Rm::Reg(RBX)))),
            ("neg r10d", &[0x41, 0xf7, 0xda], bytes(|a| a.unary(Unary::Neg, Rm::Reg(R10)))),
            ("div ecx", &[0xf7, 0xf1], bytes(|a| a.unary(Unary::Div, Rm::Reg(RCX)))),
            ("idiv ecx", &[0xf7, 0xf9], bytes(|a| a.unary(Unary::Idiv, Rm::Reg(RCX)))),
            ("imul ebx, esi", &[0x0f, 0xaf, 0xde], bytes(|a| a.imul(Long, RBX, Rm::Reg(RSI)))),
            ("imul ebx, dword ptr [rbp + 48]", &[0x0f, 0xaf, 0x5d, 0x30], bytes(|a| a.imul(Long, RBX, mem(RBP, 48)))),
            ("imul rbx, rdx", &[0x48, 0x0f, 0xaf, 0xda], bytes(|a| a.imul(Quad, RBX, Rm::Reg(RDX)))),
            ("imul r9d, r10d", &[0x45, 0x0f, 0xaf, 0xca], bytes(|a| a.imul(Long, R9, Rm::Reg(R10)))),
            ("bsr ecx, esi", &[0x0f, 0xbd, 0xce], bytes(|a| a.bsr(RCX, Rm::Reg(RSI)))),
            ("bsr ecx, dword ptr [rbp + 48]", &[0x0f, 0xbd, 0x4d, 0x30], bytes(|a| a.bsr(RCX, mem(RBP, 48)))),
            ("bswap ebx", &[0x0f, 0xcb], bytes(|a| a.bswap(RBX))),
            ("bswap r9d", &[0x41, 0x0f, 0xc9], bytes(|a| a.bswap(R9))),
            ("cmove ecx, edx", &[0x0f, 0x44, 0xca], bytes(|a| a.cmov(E, RCX, Rm::Reg(RDX)))),
            ("cmovne r8d, ebx", &[0x44, 0x0f, 0x45, 0xc3], bytes(|a| a.cmov(NE, R8, Rm::Reg(RBX)))),
            ("seto al", &[0x0f, 0x90, 0xc0], bytes(|a| a.setcc(O, Rm::Reg(RAX)))),
            ("setb cl", &[0x0f, 0x92, 0xc1], bytes(|a| a.setcc(B, Rm::Reg(RCX)))),
            ("sete byte ptr [rbp + 129]", &[0x0f, 0x94, 0x85, 0x81, 0x00, 0x00, 0x00], bytes(|a| a.setcc(E, mem(RBP, 129)))),
            ("bt eax, 8", &[0x0f, 0xba, 0xe0, 0x08], bytes(|a| a.bt(Rm::Reg(RAX), 8))),
            ("bt esi, 31", &[0x0f, 0xba, 0xe6, 0x1f], bytes(|a| a.bt(Rm::Reg(RSI), 31))),
            ("bt r12d, 3", &[0x41, 0x0f, 0xba, 0xe4, 0x03], bytes(|a| a.bt(Rm::Reg(R12), 3))),
            ("bt dword ptr [rbp + 48], 5", &[0x0f, 0xba, 0x65, 0x30, 0x05], bytes(|a| a.bt(mem(RBP, 48), 5))),
            ("cdq", &[0x99], bytes(|a| a.cdq())),
            ("lahf", &[0x9f], bytes(|a| a.lahf())),
            ("sahf", &[0x9e], bytes(|a| a.sahf())),
            ("cmc", &[0xf5], bytes(|a| a.cmc())),
            ("stc", &[0xf9], bytes(|a| a.stc())),
            ("cmp al, 129", &[0x3c, 0x81], bytes(|a| a.cmp_al(0x81))),
            ("or ah, cl", &[0x08, 0xcc], bytes(|a| a.or_ah_cl())),
            ("push rbp", &[0x55], bytes(|a| a.push(RBP))),
            ("push r15", &[0x41, 0x57], bytes(|a| a.push(R15))),
            ("pop rbx", &[0x5b], bytes(|a| a.pop(RBX))),
            ("pop r12", &[0x41, 0x5c], bytes(|a| a.pop(R12))),
            ("ret", &[0xc3], bytes(|a| a.ret())),
            ("jmp rdx", &[0xff, 0xe2], bytes(|a| a.jmp_rm(Rm::Reg(RDX)))),
            ("jmp qword ptr [rdx + 8]", &[0xff, 0x62, 0x08], bytes(|a| a.jmp_rm(mem(RDX, 8)))),
            ("call qword ptr [rdi + 56]", &[0xff, 0x57, 0x38], bytes(|a| a.call_rm(mem(RDI, 56)))),
            ("{disp32} jmp next; next:", &[0xe9, 0x00, 0x00, 0x00, 0x00], bytes(|a| { let l = a.label(); a.jmp(l); a.bind(l) })),
            ("{disp32} jne next; next:", &[0x0f, 0x85, 0x00, 0x00, 0x00, 0x00], bytes(|a| { let l = a.label(); a.jcc(NE, l); a.bind(l) })),
            ("ja next; .skip 256; next:", &[0x0f, 0x87, 0x00, 0x01, 0x00, 0x00], bytes(|a| { let l = a.label(); a.jcc(A, l); a.emit(&[0x90; 256]); a.bind(l) })[..6].to_vec()),
            ("back: .skip 256; jl back", &[0x0f, 0x8c, 0xfa, 0xfe, 0xff, 0xff], bytes(|a| { let l = a.label(); a.bind(l); a.emit(&[0x90; 256]); a.jcc(L, l) })[256..].to_vec()),
            ("jmp next; .skip 0x1ffb; next:", &[0xe9, 0xfb, 0x1f, 0x00, 0x00], bytes(|a| a.jmp_to(0x3000))),
            ("back: .skip 0x800; call back", &[0xe8, 0xfb, 0xf7, 0xff, 0xff], bytes(|a| a.call_to(0x800))),
        ];
        for (text, expected, bytes) in table {
            assert_eq!(bytes, expected, "{text}");
        }
    }
}
