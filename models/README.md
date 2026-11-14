The GitHub workflow downloads the Qwen GGUF model into this folder during the build, and buildozer
packs it into the APK. Don't commit .gguf files here - GitHub rejects files over 100 MB.
