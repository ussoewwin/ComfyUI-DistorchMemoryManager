## Summary

- Added back the missing `DisTorchPurgeVRAMV2` node that v1.10 accidentally omitted.
- Confirmed the node now ships inside the release ZIP and works after reinstall.

## Details

During v1.10 I accidentally deleted the `DisTorchPurgeVRAMV2` implementation from `__init__.py`.  
This hotfix re-adds the class, restores the node mappings, and rebuilds the release package so the node loads correctly when users reinstall.  
All work was executed via CLI (`git commit`, `git push`, `gh release create`).

