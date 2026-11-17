//! Random short programs, with the three engines of the test machine.

use super::suite::{
    CODE, Case, Insn, Program, Trio, cases, random_cpu, random_periods, t16, wide,
};
use super::{ARM_TEMPLATES, Random, WIDE_TEMPLATES};

const NOP: u32 = 0xbf00;

fn is_it(insn: &Insn) -> bool {
    insn.size == 2 && insn.raw & 0xff00 == 0xbf00 && insn.raw & 0xf != 0
}

pub fn random_thumb(random: &mut Random, at: u32, length: u32) -> Program {
    let mut insns = Vec::new();
    for _ in 0..length {
        if random.below(8) == 0 {
            insns.push(t16(0xbf00 | random.below(14) << 4 | (1 + random.below(15))));
        }
        if random.below(2) == 0 {
            let value = random.next() & 0xffff;
            // A random halfword of the 32-bit groups would take the next one as its second half.
            insns.push(t16(if value >> 11 >= 0x1d {
                value & 0x7fff
            } else {
                value
            }));
            continue;
        }
        let (fixed, mask, same_rm) =
            WIDE_TEMPLATES[random.below(WIDE_TEMPLATES.len() as u32) as usize];
        let mut raw = fixed | random.next() << 16 & mask | random.next() & mask & 0xffff;
        if same_rm {
            raw = raw & !0xf_0000 | (raw & 0xf) << 16;
        }
        insns.push(wide(raw));
    }
    // The stop after the program is not in an IT block, or else the executor does not
    // always run it. An IT in an IT block starts a new block there.
    if insns.iter().any(is_it) {
        insns.extend([t16(NOP); 4]);
    }
    Program {
        thumb: true,
        at,
        insns,
    }
}

pub fn random_arm(random: &mut Random, at: u32, length: u32) -> Program {
    let insns = (0..length)
        .map(|_| {
            let (fixed, mask) = ARM_TEMPLATES[random.below(ARM_TEMPLATES.len() as u32) as usize];
            wide(fixed | (random.next() << 16 | random.next() & 0xffff) & mask)
        })
        .collect();
    Program {
        thumb: false,
        at,
        insns,
    }
}

#[test]
fn random_blocks_run_the_same_with_each_engine() {
    let mut trio = Trio::new();
    cases("random_blocks_run_the_same_with_each_engine", 0x51ed_270b_0f3a_9a11, 150_000, |random| {
        let thumb = random.below(4) != 0;
        let mut case = Case::new(random, thumb);
        let length = 1 + random.below(6);
        // Some programs end at the last bytes of a page, some go across the page end.
        let at = CODE
            + match random.below(8) {
                0 => 0x1000 - 2 * random.below(12),
                1 => 0x1000 - 4 * random.below(6),
                _ => random.below(64) * 4,
            };
        let program = if thumb {
            random_thumb(random, at, length)
        } else {
            random_arm(random, at & !3, length)
        };
        case.cpu = random_cpu(random);
        case.load(&program);
        case.enter(&program);
        case.periods = random_periods(random);
        case.until = 1 + u64::from(random.below(24));
        trio.run(&case).map(|_| ())
    });
}
