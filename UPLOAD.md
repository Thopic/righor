How to send off a new release, cause it's hell.

- change the version number in Cargo.toml
- create a new tag `git tag v#version_number`
- commit, push, push tag

- change the version number in Cargo.toml
- commit and push.
- create a new tag with the version number `vX.X.X`
- push the tag `git push origin vX.X.X`
- cargo login
- cargo publish --dry-run
- cargo publish
