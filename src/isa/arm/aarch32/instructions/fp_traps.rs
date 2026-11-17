//! Predicate/access admission precedes the optional FP destination transaction.
use super::*;
impl<'a, M: ArmMemory> Executor<'a, M> {
    pub fn execute(&mut self, insn: &DecodedInsn) -> ExecResult {
        self.cpu.instruction_size = insn.size;
        if !self.condition_matches(insn) {
            return ExecResult::Continue;
        }
        if let Some(fault) = self.fp_access_fault(insn) {
            return fault;
        }
        let pending = crate::isa::arm::aarch32::fp_traps::Transaction::begin(self.cpu, insn);
        let result = self.execute_admitted(insn);
        match pending {
            Some(pending) => pending.finish(self.cpu, result),
            None => result,
        }
    }

    /// Runs `insn` as `execute` does after its FP access checks, and with no
    /// FP exception transaction. For a caller that saw `execute` complete
    /// this instruction with the same values of the FP controls and with no
    /// trapped FP exception enabled: the checks give the same answer again.
    pub fn execute_admitted_again(&mut self, insn: &DecodedInsn) -> ExecResult {
        self.cpu.instruction_size = insn.size;
        if !self.condition_matches(insn) {
            return ExecResult::Continue;
        }
        self.execute_admitted(insn)
    }
}
