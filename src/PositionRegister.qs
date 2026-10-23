namespace Tunneling {
    import Std.Measurement.MeasureInteger;
    operation SampleUniformPosition(nQubits : Int) : Int {
        if nQubits < 1 or nQubits > 12 {
            fail "Demo supports 1 to 12 qubits.";
        }
        use position = Qubit[nQubits];
        for q in position {
            H(q);
        }
        return MeasureInteger(position);
    }

    operation Main() : Int {
        return SampleUniformPosition(3);
    }
}
