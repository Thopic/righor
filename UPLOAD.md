How to send off a new release, cause it's hell.

First, righor:
- change the version number in Cargo.toml
- commit and push.
- create a new tag with the version number `vX.X.X`
- push the tag `git push origin vX.X.X`
- cargo login
- cargo publish --dry-run
- cargo publish

Then righor-py:
- Change the version number in Cargo.toml **and pyproject.toml**, they need to be the exact same.
- commit and push.
- create a new tag with the version number `vX.X.X`
- push the tag `git push origin vX.X.X`
- the github CI should step up, wait a bit and check how it's doing.
