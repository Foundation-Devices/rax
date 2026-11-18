//! Loads and stores against a model that follows the pseudocode of the ARM
//! architecture manual: each addressing mode with its writeback, the forms with many
//! registers, the loads of the PC, and the alignment rules.

use super::Random;
use super::machine::{RAM_BASE, Stop};
use super::suite::{CODE, Case, DATA, Insn, Program, Trio, cases, random_cpu, t16, wide};

struct Model {
    r: [u32; 16],
    thumb: bool,
    ram: Vec<u8>,
    /// The instruction gives an alignment fault and changes nothing.
    fault: bool,
}

impl Model {
    fn load(&self, address: u32, size: u32) -> u32 {
        (0..size).fold(0, |value, index| {
            let byte = self.ram[(address.wrapping_add(index) - RAM_BASE) as usize];
            value | u32::from(byte) << (8 * index)
        })
    }

    fn store(&mut self, address: u32, size: u32, value: u32) {
        for index in 0..size {
            self.ram[(address.wrapping_add(index) - RAM_BASE) as usize] =
                (value >> (8 * index)) as u8;
        }
    }

    /// LoadWritePC: bit 0 of the value selects the instruction set.
    fn branch(&mut self, value: u32) {
        self.thumb = value & 1 != 0;
        self.r[15] = value & !1;
    }
}

type Apply = Box<dyn Fn(&mut Model)>;

struct Instruction {
    arm: Option<u32>,
    thumb: Option<Insn>,
    /// The register with the address, and one with a small offset.
    base: u32,
    index: Option<u32>,
    /// The instruction has a defined result only at a multiple of 4.
    aligned: bool,
    apply: Apply,
}

fn distinct<const N: usize>(random: &mut Random, limit: u32) -> [u32; N] {
    let mut registers = [0; N];
    for index in 0..N {
        registers[index] = loop {
            let register = random.below(limit);
            if !registers[..index].contains(&register) {
                break register;
            }
        };
    }
    registers
}

/// The size of the access and true for a load with the sign: byte, halfword, word,
/// signed byte, signed halfword.
fn width(kind: u32) -> (u32, bool) {
    [(1, false), (2, false), (4, false), (1, true), (2, true)][kind as usize]
}

fn extend(value: u32, size: u32, signed: bool) -> u32 {
    match (size, signed) {
        (1, true) => value as i8 as u32,
        (2, true) => value as i16 as u32,
        _ => value,
    }
}

/// One register at `Rn + offset`, with the index mode of P and W.
#[allow(clippy::too_many_arguments)]
fn single(model: &mut Model, load: bool, kind: u32, rt: u32, rn: u32, offset: u32, p: u32, w: u32) {
    let (size, signed) = width(kind);
    let base = model.r[rn as usize];
    let with_offset = base.wrapping_add(offset);
    let address = if p == 1 { with_offset } else { base };
    if load {
        let value = extend(model.load(address, size), size, signed);
        if p == 0 || w == 1 {
            model.r[rn as usize] = with_offset;
        }
        model.r[rt as usize] = value;
    } else {
        model.store(address, size, model.r[rt as usize]);
        if p == 0 || w == 1 {
            model.r[rn as usize] = with_offset;
        }
    }
}

fn signed_offset(u: u32, offset: u32) -> u32 {
    if u == 1 {
        offset
    } else {
        offset.wrapping_neg()
    }
}

/// P, U and W for an offset, an access before a writeback, and one after it.
fn index_mode(random: &mut Random) -> (u32, u32, u32) {
    let (p, w) = [(1, 0), (1, 1), (0, 0)][random.below(3) as usize];
    (p, random.below(2), w)
}

fn single_register(random: &mut Random) -> Instruction {
    let [rt, rn, rm] = distinct(random, 13);
    let load = random.below(2);
    let kind = if load == 1 {
        random.below(5)
    } else {
        random.below(3)
    };
    let (p, u, w) = index_mode(random);
    let imm = random.below(256);
    // The A32 and T32 fields of the sizes.
    let arm_half = [0, 0xb0, 0, 0xd0, 0xf0][kind as usize];
    let thumb_size = [
        0xf800_0000u32,
        0xf820_0000,
        0xf840_0000,
        0xf900_0000,
        0xf920_0000,
    ][kind as usize];
    let puw = p << 24 | u << 23 | w << 21;
    match random.below(4) {
        // An immediate offset.
        0 | 1 => {
            let arm = match kind {
                0 | 2 => {
                    0x0400_0000
                        | puw
                        | u32::from(kind == 0) << 22
                        | load << 20
                        | rn << 16
                        | rt << 12
                        | imm
                }
                _ => {
                    0x0040_0000
                        | puw
                        | load << 20
                        | rn << 16
                        | rt << 12
                        | (imm >> 4) << 8
                        | arm_half
                        | imm & 15
                }
            };
            // T32 has no negative offset with no writeback in this form but with 8 bits.
            let thumb = thumb_size
                | load << 20
                | rn << 16
                | rt << 12
                | 0x800
                | p << 10
                | u << 9
                | ((1 - p) | w) << 8
                | imm;
            let thumb = (!(p == 1 && w == 0 && u == 1)).then_some(wide(thumb));
            Instruction {
                arm: Some(arm),
                thumb,
                base: rn,
                index: None,
                aligned: false,
                apply: Box::new(move |model| {
                    single(model, load == 1, kind, rt, rn, signed_offset(u, imm), p, w);
                }),
            }
        }
        // The T32 offset of 12 bits, and the A32 one.
        2 => {
            let imm = random.below(0x400);
            let arm = (kind == 0 || kind == 2).then_some(
                0x0580_0000 | u32::from(kind == 0) << 22 | load << 20 | rn << 16 | rt << 12 | imm,
            );
            Instruction {
                arm,
                thumb: Some(wide(
                    thumb_size | 0x0080_0000 | load << 20 | rn << 16 | rt << 12 | imm,
                )),
                base: rn,
                index: None,
                aligned: false,
                apply: Box::new(move |model| single(model, load == 1, kind, rt, rn, imm, 1, 0)),
            }
        }
        // A register offset: A32 with each shift for the word and the byte, T32 with LSL.
        _ => {
            // The offset register has a small value: these shifts keep it small.
            let shift = random.below(4);
            let amount = match shift {
                0 => random.below(5),
                3 => 0,
                _ => random.below(32),
            };
            let left = random.below(4);
            let arm = match kind {
                0 | 2 => {
                    0x0600_0000
                        | puw
                        | u32::from(kind == 0) << 22
                        | load << 20
                        | rn << 16
                        | rt << 12
                        | amount << 7
                        | shift << 5
                        | rm
                }
                _ => puw | load << 20 | rn << 16 | rt << 12 | 0x90 | arm_half & 0x60 | rm,
            };
            let thumb = thumb_size | load << 20 | rn << 16 | rt << 12 | left << 4 | rm;
            if random.below(2) == 0 {
                return Instruction {
                    arm: None,
                    thumb: Some(wide(thumb)),
                    base: rn,
                    index: Some(rm),
                    aligned: false,
                    apply: Box::new(move |model| {
                        let offset = model.r[rm as usize] << left;
                        single(model, load == 1, kind, rt, rn, offset, 1, 0);
                    }),
                };
            }
            Instruction {
                arm: Some(arm),
                thumb: None,
                base: rn,
                index: Some(rm),
                aligned: false,
                apply: Box::new(move |model| {
                    let value = model.r[rm as usize];
                    let offset = match (kind, shift, amount) {
                        (1 | 3 | 4, ..) => value,
                        (_, 0, _) => value << amount,
                        (_, 1, 0) => 0,
                        (_, 1, _) => value >> amount,
                        (_, 2, 0) => (value as i32 >> 31) as u32,
                        (_, 2, _) => (value as i32 >> amount) as u32,
                        // RRX: the carry flag is clear in these cases.
                        (_, _, 0) => value >> 1,
                        _ => value.rotate_right(amount),
                    };
                    single(
                        model,
                        load == 1,
                        kind,
                        rt,
                        rn,
                        signed_offset(u, offset),
                        p,
                        w,
                    );
                }),
            }
        }
    }
}

/// LDRD and STRD: an address that is not a multiple of 4 gives an alignment fault.
fn pair(random: &mut Random) -> Instruction {
    let [rn, other] = distinct(random, 13);
    let rt = loop {
        let rt = random.below(6) * 2;
        if rt != rn && rt + 1 != rn {
            break rt;
        }
    };
    // T32 takes each second register.
    let rt2 = if other == rt || other == rn {
        rt + 1
    } else {
        other
    };
    let load = random.below(2);
    let (p, u, w) = index_mode(random);
    let imm = random.below(64) * 4;
    let apply = move |rt2: u32, offset: u32| -> Apply {
        Box::new(move |model| {
            let base = model.r[rn as usize];
            let with_offset = base.wrapping_add(signed_offset(u, offset));
            let address = if p == 1 { with_offset } else { base };
            if address & 3 != 0 {
                model.fault = true;
                return;
            }
            if load == 1 {
                model.r[rt as usize] = model.load(address, 4);
                model.r[rt2 as usize] = model.load(address + 4, 4);
            } else {
                model.store(address, 4, model.r[rt as usize]);
                model.store(address + 4, 4, model.r[rt2 as usize]);
            }
            if p == 0 || w == 1 {
                model.r[rn as usize] = with_offset;
            }
        })
    };
    if random.below(2) == 0 {
        let raw = 0xe840_0000
            | p << 24
            | u << 23
            | ((1 - p) | w) << 21
            | load << 20
            | rn << 16
            | rt << 12
            | rt2 << 8
            | (imm / 4);
        return Instruction {
            arm: None,
            thumb: Some(wide(raw)),
            base: rn,
            index: None,
            aligned: false,
            apply: apply(rt2, imm),
        };
    }
    let imm = imm & 0xff;
    let raw = 0x0040_00d0
        | p << 24
        | u << 23
        | w << 21
        | rn << 16
        | rt << 12
        | (imm >> 4) << 8
        | (1 - load) << 5
        | imm & 15;
    Instruction {
        arm: Some(raw),
        thumb: None,
        base: rn,
        index: None,
        aligned: false,
        apply: apply(rt + 1, imm),
    }
}

/// LDM and STM in the four orders, PUSH and POP. The PC is in some lists of loads.
fn multiple(random: &mut Random) -> Instruction {
    let rn = if random.below(3) == 0 {
        13
    } else {
        random.below(13)
    };
    let load = random.below(2);
    let writeback = random.below(2);
    let mut mask = random.next() & 0x5fff;
    if load == 1 && random.below(4) == 0 {
        mask |= 0x8000;
    }
    // The base is not in the list with a writeback: its value has no definition then.
    mask &= !(writeback << rn);
    if mask.count_ones() < 2 {
        mask |= if rn < 2 { 0x30 } else { 3 };
    }
    let count = mask.count_ones();
    // Increment or decrement, and before or after.
    let (up, before) = (random.below(2), random.below(2));
    let apply: Apply = Box::new(move |model| {
        let base = model.r[rn as usize];
        let low = match (up, before) {
            (1, 0) => base,
            (1, _) => base.wrapping_add(4),
            (_, 0) => base.wrapping_sub(4 * count).wrapping_add(4),
            _ => base.wrapping_sub(4 * count),
        };
        if base & 3 != 0 {
            model.fault = true;
            return;
        }
        let mut address = low;
        let mut target = None;
        for register in 0..16 {
            if mask >> register & 1 == 0 {
                continue;
            }
            if load == 0 {
                model.store(address, 4, model.r[register]);
            } else if register == 15 {
                target = Some(model.load(address, 4));
            } else {
                model.r[register] = model.load(address, 4);
            }
            address = address.wrapping_add(4);
        }
        if writeback == 1 {
            model.r[rn as usize] = if up == 1 {
                base.wrapping_add(4 * count)
            } else {
                base.wrapping_sub(4 * count)
            };
        }
        if let Some(target) = target {
            model.branch(target);
        }
    });
    let arm =
        0x0800_0000 | before << 24 | up << 23 | writeback << 21 | load << 20 | rn << 16 | mask;
    // T32 has the increment after and the decrement before, and not LR with the PC.
    let thumb = (up != before && mask & 0xc000 != 0xc000).then(|| {
        wide(0xe800_0000 | before << 24 | up << 23 | writeback << 21 | load << 20 | rn << 16 | mask)
    });
    Instruction {
        arm: Some(arm),
        thumb,
        base: rn,
        index: None,
        aligned: false,
        apply,
    }
}

/// The 16-bit Thumb forms.
fn narrow(random: &mut Random) -> Instruction {
    let [rt, rn, rm] = distinct(random, 8);
    let load = random.below(2);
    let imm5 = random.below(32);
    let mask = (1 + random.below(255)) & !(1 << rn) | 1 << rt;
    let count = mask.count_ones() + 1;
    let (raw, base, index, apply): (u32, u32, Option<u32>, Apply) = match random.below(8) {
        0 => (
            0x6000 | load << 11 | imm5 << 6 | rn << 3 | rt,
            rn,
            None,
            Box::new(move |model| single(model, load == 1, 2, rt, rn, imm5 * 4, 1, 0)),
        ),
        1 => (
            0x7000 | load << 11 | imm5 << 6 | rn << 3 | rt,
            rn,
            None,
            Box::new(move |model| single(model, load == 1, 0, rt, rn, imm5, 1, 0)),
        ),
        2 => (
            0x8000 | load << 11 | imm5 << 6 | rn << 3 | rt,
            rn,
            None,
            Box::new(move |model| single(model, load == 1, 1, rt, rn, imm5 * 2, 1, 0)),
        ),
        3 => (
            0x9000 | load << 11 | rt << 8 | imm5,
            13,
            None,
            Box::new(move |model| single(model, load == 1, 2, rt, 13, imm5 * 4, 1, 0)),
        ),
        4 => {
            // STR, STRH, STRB, LDRSB, LDR, LDRH, LDRB, LDRSH with a register offset.
            let operation = random.below(8);
            let (load, kind) = [
                (false, 2),
                (false, 1),
                (false, 0),
                (true, 3),
                (true, 2),
                (true, 1),
                (true, 0),
                (true, 4),
            ][operation as usize];
            (
                0x5000 | operation << 9 | rm << 6 | rn << 3 | rt,
                rn,
                Some(rm),
                Box::new(move |model| {
                    let offset = model.r[rm as usize];
                    single(model, load, kind, rt, rn, offset, 1, 0);
                }),
            )
        }
        // PUSH with LR, and POP with the PC.
        5 => (
            0xb400 | load << 11 | 1 << 8 | mask & 0xff,
            13,
            None,
            Box::new(move |model| {
                let sp = model.r[13];
                if sp & 3 != 0 {
                    model.fault = true;
                    return;
                }
                let mut address = if load == 1 {
                    sp
                } else {
                    sp.wrapping_sub(4 * count)
                };
                let low = address;
                for register in (0..8).chain([if load == 1 { 15 } else { 14 }]) {
                    if register < 8 && mask >> register & 1 == 0 {
                        continue;
                    }
                    if load == 0 {
                        model.store(address, 4, model.r[register]);
                    } else if register == 15 {
                        let target = model.load(address, 4);
                        model.branch(target);
                    } else {
                        model.r[register] = model.load(address, 4);
                    }
                    address = address.wrapping_add(4);
                }
                model.r[13] = if load == 1 { address } else { low };
            }),
        ),
        // STMIA with a writeback, and LDMIA: a writeback when the base is not in the list.
        _ => {
            let mask = if load == 1 && random.below(2) == 0 {
                mask | 1 << rn
            } else {
                mask
            };
            (
                0xc000 | load << 11 | rn << 8 | mask & 0xff,
                rn,
                None,
                Box::new(move |model| {
                    let base = model.r[rn as usize];
                    if base & 3 != 0 {
                        model.fault = true;
                        return;
                    }
                    let mut address = base;
                    for register in 0..8 {
                        if mask >> register & 1 == 0 {
                            continue;
                        }
                        if load == 1 {
                            model.r[register] = model.load(address, 4);
                        } else {
                            model.store(address, 4, model.r[register]);
                        }
                        address = address.wrapping_add(4);
                    }
                    if mask >> rn & 1 == 0 {
                        model.r[rn as usize] = address;
                    }
                }),
            )
        }
    };
    Instruction {
        arm: None,
        thumb: Some(t16(raw)),
        base,
        index,
        aligned: false,
        apply,
    }
}

/// Loads from an address near the PC, and loads of the PC.
fn with_pc(random: &mut Random) -> Instruction {
    let [rt, rn] = distinct(random, 13);
    let up = random.below(2);
    let imm = random.below(64) * 4;
    let kind = random.below(5);
    let (size, signed) = width(kind);
    let thumb_size = [
        0xf81f_0000u32,
        0xf83f_0000,
        0xf85f_0000,
        0xf91f_0000,
        0xf93f_0000,
    ][kind as usize];
    match random.below(4) {
        0 => Instruction {
            arm: Some(0x051f_0000 | up << 23 | rt << 12 | imm),
            thumb: None,
            base: rn,
            index: None,
            aligned: false,
            apply: Box::new(move |model| {
                let address = model.r[15]
                    .wrapping_add(8)
                    .wrapping_add(signed_offset(up, imm));
                model.r[rt as usize] = model.load(address, 4);
            }),
        },
        1 => Instruction {
            arm: None,
            thumb: Some(wide(thumb_size | up << 23 | rt << 12 | imm)),
            base: rn,
            index: None,
            aligned: false,
            apply: Box::new(move |model| {
                let address =
                    (model.r[15].wrapping_add(4) & !3).wrapping_add(signed_offset(up, imm));
                model.r[rt as usize] = extend(model.load(address, size), size, signed);
            }),
        },
        2 => {
            let rt = rt % 8;
            Instruction {
                arm: None,
                thumb: Some(t16(0x4800 | rt << 8 | (imm / 4))),
                base: rn,
                index: None,
                aligned: false,
                apply: Box::new(move |model| {
                    let address = (model.r[15].wrapping_add(4) & !3).wrapping_add(imm);
                    model.r[rt as usize] = model.load(address, 4);
                }),
            }
        }
        // LDR of the PC, with an offset and with a writeback after the access.
        _ => {
            let post = random.below(2);
            Instruction {
                arm: Some(0x0490_f000 | (1 - post) << 24 | rn << 16 | imm),
                thumb: Some(wide(if post == 1 {
                    0xf850_fb00 | rn << 16 | imm
                } else {
                    0xf8d0_f000 | rn << 16 | imm
                })),
                base: rn,
                index: None,
                aligned: true,
                apply: Box::new(move |model| {
                    let base = model.r[rn as usize];
                    let address = if post == 1 {
                        base
                    } else {
                        base.wrapping_add(imm)
                    };
                    let target = model.load(address, 4);
                    if post == 1 {
                        model.r[rn as usize] = base.wrapping_add(imm);
                    }
                    model.branch(target);
                }),
            }
        }
    }
}

type Generate = fn(&mut Random) -> Instruction;

const CLASSES: [(&str, Generate); 5] = [
    ("single_register", single_register),
    ("pair", pair),
    ("multiple", multiple),
    ("narrow", narrow),
    ("with_pc", with_pc),
];

fn check(trio: &mut Trio, random: &mut Random) -> Result<(), String> {
    let (name, generate) = CLASSES[random.below(CLASSES.len() as u32) as usize];
    let instruction = generate(random);
    let (insn, thumb) = match (instruction.arm, instruction.thumb) {
        (Some(arm), Some(_)) if random.below(2) == 0 => (wide(0xe000_0000 | arm), false),
        (_, Some(thumb)) => (thumb, true),
        (Some(arm), None) => (wide(0xe000_0000 | arm), false),
        (None, None) => unreachable!(),
    };
    let mut case = Case::new(random, thumb);
    case.cpu = random_cpu(random);
    case.cpu.cpsr.c = false;
    // A load of the PC takes each word as an address: one of A32 code is a multiple of 4.
    for at in (DATA + 0x2000..DATA + 0x6000).step_by(4) {
        let word = case.ram.word(at);
        if word & 1 == 0 {
            case.ram.write_words(at, &[word & !2]);
        }
    }
    for register in &mut case.cpu.regs[..15] {
        *register = random.next();
    }
    // Most addresses are multiples of 4.
    let aligned = instruction.aligned || random.below(8) != 0;
    let base =
        DATA + 0x3000 + random.below(0x400) * 4 + if aligned { 0 } else { 1 + random.below(3) };
    case.cpu.regs[instruction.base as usize] = base;
    if let Some(index) = instruction.index {
        case.cpu.regs[index as usize] = random.below(64);
    }
    let program = Program {
        thumb,
        at: CODE + 0x400 + random.below(64) * 4 + if thumb { random.below(2) * 2 } else { 0 },
        insns: vec![insn],
    };
    case.load(&program);
    case.enter(&program);
    case.until = 1;
    let mut model = Model {
        r: case.cpu.regs,
        thumb,
        ram: case.ram.bytes.clone(),
        fault: false,
    };
    (instruction.apply)(&mut model);
    if model.r[15] == program.at {
        model.r[15] = program.end();
    }
    let outcome = trio.run(&case)?;
    let (stopped, same) = match &outcome.stop {
        Some(Stop::Exception(detail)) => (
            detail.starts_with("data abort"),
            outcome.regs == case.cpu.regs && outcome.ram == case.ram.bytes,
        ),
        Some(_) => (false, false),
        None => (
            false,
            outcome.regs == model.r
                && (outcome.cpsr >> 5 & 1 != 0) == model.thumb
                && outcome.ram == model.ram,
        ),
    };
    if stopped == model.fault && same {
        return Ok(());
    }
    let at = (0..model.ram.len()).find(|&at| outcome.ram[at] != model.ram[at]);
    Err(format!(
        "{name} {:?} {:08x} : {} {:x} from {:x?} cpsr {:#x}\n  the engines: {:x?} cpsr {:#x} {:?}\n  the model:   {:x?} thumb {} fault {}\n  memory: {:x?}",
        rax::arm::Decoder::new(if thumb {
            rax::arm::ExecutionState::Thumb
        } else {
            rax::arm::ExecutionState::Aarch32
        })
        .decode(&program.bytes())
        .map(|insn| insn.mnemonic),
        insn.raw & if insn.size == 2 { 0xf800 } else { 0xfff0_0000 },
        if thumb { "Thumb" } else { "A32" },
        insn.raw,
        case.cpu.regs,
        case.cpu.cpsr.to_u32(),
        outcome.regs,
        outcome.cpsr,
        outcome.stop,
        model.r,
        model.thumb,
        model.fault,
        at.map(|at| (RAM_BASE as usize + at, outcome.ram[at], model.ram[at])),
    ))
}

#[test]
fn loads_and_stores_give_the_results_of_the_architecture() {
    let mut trio = Trio::new();
    cases(
        "loads_and_stores_give_the_results_of_the_architecture",
        0x0a11_ce5e_ed00_0008,
        60_000,
        |random| check(&mut trio, random),
    );
}
