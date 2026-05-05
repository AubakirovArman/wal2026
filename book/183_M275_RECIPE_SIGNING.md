# M275 — Recipe Signing / Tamper Detection

**Date:** 2026-04-20
**File:** `experiments/m275_recipe_signing.py`

## Purpose

Cryptographically sign recipes to detect tampering.

## Results

- Original: ✅ valid
- Tampered answer: ❌ correctly rejected
- Tampered seed: ❌ correctly rejected

## Conclusion

🎯 **Recipe signing works.** HMAC-SHA256 detects all tampering.
