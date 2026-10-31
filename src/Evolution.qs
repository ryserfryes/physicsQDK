namespace Tunneling {
    import Std.Arrays.Reversed;
    import Std.Canon.ApplyQFT;
    import Std.Canon.SwapReverseRegister;
    import Std.StatePreparation.PreparePureStateD;

    operation PrepareWavePacket(position : Qubit[], envelope : Double[], positionBitPhases : Double[]) : Unit {
        if Length(envelope) != 1 <<< Length(position) or Length(positionBitPhases) != Length(position) {
            fail "Packet dimensions must match the position register.";
        }
        PreparePureStateD(envelope, position);
        for i in 0..Length(position) - 1 {
            R1(positionBitPhases[i], position[i]);
        }
    }
    
    operation EvolutionStep(position : Qubit[], kineticBitPhases : Double[], kineticPairPhases : Double[][]) : Unit {
        let n = Length(position);
        if Length(kineticBitPhases) != n or Length(kineticPairPhases) != n {
            fail "Kinetic phase dimensions must match the position register.";
        }
        let momentum = Reversed(position);
        ApplyQFT(momentum);
        SwapReverseRegister(momentum);
        for i in 0..n - 1 {
            if Length(kineticPairPhases[i]) != n {
                fail "Kinetic pair phase dimensions must match the position register.";
            }
            R1(kineticBitPhases[i], momentum[i]);
            for j in i + 1..n - 1 {
                Controlled R1([momentum[i]], (kineticPairPhases[i][j], momentum[j]));
            }
        }
        Adjoint SwapReverseRegister(momentum);
        Adjoint ApplyQFT(momentum);
    }
}
