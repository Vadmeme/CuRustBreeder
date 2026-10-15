# CUDA Rust Plant Genetics Calculator

## Features

- Uses center sapling placement to break 50/50 ties  
- Uses sapling directionality to break 50/50 ties  
- Computes permutations, allowing it to find plants with highly complex constraints  
- Supports genes with multiple two-way ties  
- Generally able to find much simpler paths to obtain a clone in almost all cases  
- The main downside is that it does not consider saplings with non-unique inputs  
- Runs significantly faster than every other tool - on my GPU (RTX 3080), it takes around 0.8 seconds to process a full generation of 180 genes, or about 2 seconds total execution time  

## Why no GUI?

Its easier for me that way =)

## Comparison to Other Tools

A useful example that highlights the limitations of other tools is asking them to find the `GYYYGG` clone from the following input set in a single generation. Below is a table of various Rust genetics calculators I found and how they perform on this test:

```
XYYWGG  
WYYYHH  
GWGXGX  
GHGYHG  
```

| Tool | Found Gene | Win 50/50 Tie | Notes |
|---|---|---|---|
| rustbreeder.com | ⚠️ Partial | ❌ Fail | Finds in Gen2, not Gen1. This is due to a lack of support for complex ties (best overall alternative, though). |
| rustmeta.com/crossbreed | ⚠️ Partial | ❌ Fail | Finds the gene but provides no guide on how to win the tie. |
| wgn.si/genetics | ⚠️ Partial | ❌ Fail | Finds the gene but offers no tie-breaking guidance. |
| rusttips.com/rust-genetics-calculator | ❌ Excluded | ❌ Excluded | Only supports a single plant. |
| rustabona.net/tools-genetics-calculator.html | ❌ Excluded | ❌ Excluded | Only supports a single plant. |
| jaretburkett.github.io/rust-genetics-calculator | ⚠️ Partial | ❌ Fail | Finds the clone among results but provides no tie-winning path. |
| alierum.github.io | ⚠️ Partial | ❌ Fail | Finds the clone among results but provides no tie-winning path. |
| pr0xs.github.io/Rust-Genetics-Lab | ⚠️ Partial | ❌ Fail | Finds the clone among results but provides no tie-winning path. |
| zapishnyi.github.io Rust Genetics | ❌ Fail | ❌ Fail | Does not find the clone at all. |
| github.com/timginter/rust-gene-calculator | ⚠️ Partial | ❌ Fail | Reimplementation of wgn.si/genetics with the same limitations. |
