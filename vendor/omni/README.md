# Vendored llama.cpp-omni sources

The [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) fork of llama.cpp (see root README
for more info) includes many utilities for properly encoding/decoding input and output for
MiniCPM-o 4.5 model prompts.

This directory contains code lifted from the llama.cpp-omni repository (at commit 0788f99).

Most files remain identical to the upstream source, but a notable exception is `omni-tts.h`/`.cpp`.
This file contains TTS-specific extracts from the upstream `omni.h`/`.cpp` sources - other code
from that translation unit is eschewed in favour of a redesign using modern C++ in the main project
sources.