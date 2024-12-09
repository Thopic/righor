How to send off a new release, cause it's hell.

- change the version number in Cargo.toml
- create a new tag `git tag v#version_number`
- commit, push, push tag


First, righor:
- change the version number in Cargo.toml
- commit and push.
- create a new tag with the version number `vX.X.X`
- push the tag `git push origin vX.X.X`
- cargo login
- cargo publish --dry-run
- cargo publish

Then righor-py:
- Change the righor call in Cargo.toml (put version nb)
- Change the version number in Cargo.toml **and pyproject.toml**, they need to be the exact same.
- commit and push.
- create a new tag with the version number `vX.X.X`
- push the tag `git push origin vX.X.X`
- the github CI should step up, wait a bit and check how it's doing.
