# macOS (Apple Silicon) 部署

在 Mac（arm64）上编译、跑通并验证本框架的完整步骤与注意事项。Linux / Windows 的常规构建见主 README，本文只补充 macOS 特有的部分。

## 📑 目录

- [依赖安装](#依赖安装)
- [ONNX Runtime：必须用官方 C SDK 包](#onnx-runtime必须用官方-c-sdk-包)
- [构建](#构建)
- [OpenCV 4 与 5 的差异](#opencv-4-与-5-的差异)
- [加速：CoreML EP](#加速coreml-ep)
- [Python 参考环境与 exFAT 卷的坑](#python-参考环境与-exfat-卷的坑)
- [验证清单](#验证清单)

## 依赖安装

```bash
brew install cmake pkg-config opencv
```

`pkg-config` 是必需的：`FindOpenCV` 走 `pkg-config` 查询，缺失时 CMake 报找不到 OpenCV，而不是给出可读的错误。

## ONNX Runtime：必须用官方 C SDK 包

pip 装的 `onnxruntime` 只有 Python 扩展，**不带 C/C++ 头文件与独立的 `libonnxruntime`**，不能用给 C++ 侧链接。从官方 release 下载 osx-arm64 预编译包：

```bash
curl -L -o /tmp/ort.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-osx-arm64-1.30.0.tgz
mkdir -p ~/download && tar xzf /tmp/ort.tgz -C ~/download
export ONNXRUNTIME_DIR=~/download/onnxruntime-osx-arm64-1.30.0
```

`cmake/FindONNXRuntime.cmake` 会查 `$ENV{ONNXRUNTIME_DIR}/include` 与 `lib`，也可以用 `-DONNXRUNTIME_DIR=...` 传。

## 构建

```bash
mkdir build && cd build
cmake .. -DWITH_EXAMPLES=ON -DWITH_TESTS=ON
make -j8
./tests/test_yolo
```

macOS 没有 `nproc`，主 README 里的 `make -j$(nproc)` 在这里会失败；取核数用 `sysctl -n hw.ncpu`。

`libonnxruntime.dylib` 装在上一步解压的目录下，二进制用 `@rpath` 引用它。从别处运行示例时如果报 dylib 找不到，用 `otool -L ./detect_image` 看引用路径，再用 `install_name_tool` 或 `DYLD_LIBRARY_PATH` 指过去。Apple Silicon 上首次跑会经过 Gatekeeper 签名校验，本地编译的产物不受影响，但**不要**对构建目录做 `chmod -R` / 删除签名，否则会被系统杀掉。

## OpenCV 4 与 5 的差异

brew 现在默认装 5.x，`contourArea` / `rotatedRectangleIntersection` 从 `imgproc` 移到了 `opencv2/geometry.hpp`，直接编译会报 `no member named 'rotatedRectangleIntersection' in namespace 'cv'`。`src/process/postprocess/postprocess_core.cpp` 已按 `CV_VERSION_MAJOR` 条件包含，4.x / 5.x 都能编译。

要与 Linux 侧保持一致可以 `brew install opencv@4`（keg-only，需要 `export OpenCV_DIR=$(brew --prefix opencv@4)/lib/cmake/opencv4`）。

## 加速：CoreML EP

Mac 上的 GPU / ANE 走 CoreML EP：

```cpp
Config::custom_config = "ep=coreml;ModelFormat=MLProgram;MLComputeUnits=CPUAndGPU";
```

`yolo11n@640` 实测 **5.3 ms/帧**，CPU（4 线程）24.6 ms。两个必须知道的点：

- **不加 `ModelFormat=MLProgram`，每次加载要多付约 4 秒** CoreML 编译，进程短的场景（跑一张图就退出）比 CPU 慢 40 倍。
- **`device_id` 会让 CoreML 注册失败并静默回退 CPU**，`src/core/onnxruntime_backend.cpp` 已改成只在显式指定非 0 设备时才透传。

`xnnpack` 不在官方 osx-arm64 包里，选择它会打印原因并回退 CPU。

完整的 CoreML 基准表、如何确认真的跑在 CoreML 上、三个坑以及语义分割头为何不适合 CoreML —— 见 **[docs/backends.md](backends.md)** 的「CoreML EP（macOS / Apple Silicon）」一节。

## Python 参考环境与 exFAT 卷的坑

`tools/` 的精度比对需要一个 Python 环境：

```bash
python3 -m venv ~/.venvs/yolo-onnx
~/.venvs/yolo-onnx/bin/pip install onnx onnxruntime numpy opencv-python
```

**不要把 venv 建在项目所在的卷上**，如果项目放在 exFAT / NTFS 这类非 HFS+/APFS 卷上，macOS 会为每个文件生成 `._xxx` AppleDouble 伴生文件；`site.py` 会把 `._distutils-precedence.pth` 当文本读取，直接以 `UnicodeDecodeError` 崩掉，Python 环境建不起来也进不去。venv 与 `tools/` 比对的输出目录都建议放本地磁盘（如 `~/.venvs`、`/tmp`）。

同理，这类卷上的 `._*` 文件会污染 `git status`，`.gitignore` 已忽略；`.git/objects/pack/` 里若混进 `._pack-*.idx`，git 命令会打印 `non-monotonic index` 错误（不影响结果，`find .git -name '._*' -delete` 可清掉）。

## 验证清单

在本机跑通的内容：

- `tests/test_yolo` —— 84 个单元测试全通过
- 5 个示例（detect / segment / pose / sem / obb）+ `tools/dump_json` 均能编译运行
- 精度比对（`dump_json` → `ref_onnx.py` → `compare.py`）：detect 覆盖 v5 / yolox / v11 / v26 × bus / zidane，动态 shape @640 与 @320，segment ×3（框 IoU 0.999），sem（像素一致率 0.9939）全部 PASS
- 已知未覆盖：pose 没有 Python 参考实现，只能跑 C++ 侧自检；obb 缺少 DOTA 类倾斜框样本，比对结果为空集

工具链本身的用法与历史 bug 复盘见 **[docs/accuracy.md](accuracy.md)**。
