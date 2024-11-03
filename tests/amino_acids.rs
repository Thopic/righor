use anyhow::Result;

use righor::{shared::DnaLike, AminoAcid};

#[test]
fn amino_acids_test() -> Result<()> {
    // Basic test, ne inference with a real model and a real sequence
    // Just check that nothing panic or return an error.
    // Note: this rely on the presence of data files, so it may
    // fail if the data files are not present
    let amino_acid_str = "CAFREW";
    let amino_acid = AminoAcid::from_string(amino_acid_str)?;
    let seq = DnaLike::from_amino_acid(amino_acid.clone());
    assert!(seq.translate()? == amino_acid.clone());

    assert!(seq.extract_subsequence(3, 6).translate()? == AminoAcid::from_string("A")?);
    assert!(seq.extract_subsequence(3, 9).translate()? == AminoAcid::from_string("AF")?);
    assert!(seq.extract_subsequence(3, 18).translate()? == AminoAcid::from_string("AFREW")?);

    Ok(())
}
