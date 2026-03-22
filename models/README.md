Local model assets live under this directory at deploy time.

Expected layout:

- `models/sam3/sam3.pt`
- `models/sam3/bpe_simple_vocab_16e6.txt.gz`
- `models/sam3-official/sam3/...`
- `models/RMBG-2.0/...`

Large weights are intentionally ignored by git. To populate this directory on a machine, run:

```bash
bash script/prepare_local_models.sh
```
