# Redeem Codes

Default points-code files:

- `points_10.txt`
- `points_50.txt`
- `points_100.txt`

Format:

- One token per line
- Blank lines and lines starting with `#` are ignored
- Tokens are matched case-insensitively by the backend

Generate a fresh batch with:

```bash
python script/generate_redeem_codes.py --count 200
```
