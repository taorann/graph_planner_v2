# Windows 远程 Docker 与 SWE-bench 托管指南

本文档说明如何在 Windows 11/10 PC 上为 DeepSWE/RepoEnv 准备远程 Docker 宿主，并解答常见问题，例如在 PowerShell 中出现 `openssl` 未被识别的报错。

## 1. 基础准备

1. 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 并使用 **Docker Personal** 账号登录。
2. 启用 WSL2 与虚拟化功能。
3. 以管理员身份打开 PowerShell 或 Windows Terminal。
4. 安装 OpenSSL 工具集（详见下一节），用于生成 Docker TLS 证书。

## 2. 安装 OpenSSL（解决 `openssl` 未被识别）

如果在 `C:\docker-certs>` 目录执行 `openssl genrsa` 出现 `无法将“openssl”项识别为 cmdlet`，说明系统中尚未安装或未配置 OpenSSL。

### 方案 A：使用 winget 安装

```powershell
winget install --id=ShiningLight.OpenSSL.Light --source=winget
```

安装完成后重新打开终端，让 `C:\Program Files\OpenSSL-Win64\bin` 自动加入 `PATH`。

### 方案 B：使用 Chocolatey

1. 先安装 [Chocolatey](https://chocolatey.org/install)。
2. 在管理员 PowerShell 中运行：
   ```powershell
   choco install openssl.light
   ```

### 方案 C：直接下载安装包

1. 访问 [Shining Light Productions](https://slproweb.com/products/Win32OpenSSL.html)。
2. 下载 `Win64 OpenSSL Light` 安装包并执行安装。
3. 在安装向导中勾选 *Copy OpenSSL DLLs to The Windows system directory*，或手动将安装目录的 `bin` 路径添加到系统环境变量 `PATH`。

完成安装后，在新的 PowerShell 会话中执行：

```powershell
openssl version
```

若看到 OpenSSL 版本号，说明命令已可用。

## 3. 生成 TLS 证书

1. 创建证书目录：
   ```powershell
   mkdir C:\docker-certs
   cd C:\docker-certs
   ```
2. 按顺序执行以下命令生成 CA、服务器与客户端证书（`<公网IP>`、`<局域网IP>`、`<域名>` 需按实际情况替换）：
   ```powershell
   openssl genrsa -aes256 -out ca-key.pem 4096
   openssl req -new -x509 -days 3650 -key ca-key.pem -sha256 -out ca.pem

   openssl genrsa -out server-key.pem 4096
   openssl req -subj "/CN=windows-docker" -sha256 -new -key server-key.pem -out server.csr
   # 用你的实际地址替换尖括号中的占位符：
   # - `<公网IP>`：在浏览器访问 https://ifconfig.me/ 或 https://ipinfo.io/ 获取；
   # - `<局域网IP>`：在 PowerShell 中执行 `ipconfig`，取正在使用网卡的 IPv4 地址；
   # - `<域名>`：如果已有指向此主机的 DNS 记录（例如 example.com），填域名；没有域名可删除整段 `,DNS:<域名>`；
   # 可以按照需要增删 `IP:` / `DNS:` 条目，多个条目之间用英文逗号分隔。
   # 示例：使用 ASCII 编码写入 extfile.cnf，避免 PowerShell 默认的 UTF-16 编码导致 OpenSSL 报 “missing equal sign”
   Set-Content -Encoding ascii -Path extfile.cnf -Value 'subjectAltName = IP:127.0.0.1,IP:185.150.138.42,IP:172.18.64.1'
   # 如果你还需要加入域名，可在末尾追加 `,DNS:example.com` 或替换为自己的条目：
   Set-Content -Encoding ascii -Path extfile.cnf -Value 'subjectAltName = IP:127.0.0.1,IP:<公网IP>,IP:<局域网IP>,DNS:<域名>'
   # 可通过 `Get-Content extfile.cnf` 检查文件内容是否与期望一致。
   openssl x509 -req -days 3650 -sha256 -in server.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out server-cert.pem -extfile extfile.cnf

   openssl genrsa -out key.pem 4096
   openssl req -subj "/CN=agent-client" -new -key key.pem -out client.csr
   # 同样使用 ASCII 编码写入客户端用途扩展，避免 UTF-16 导致的 "missing equal sign" 错误。
   Set-Content -Encoding ascii -Path extfile-client.cnf -Value 'extendedKeyUsage = clientAuth'
   # 可选：使用 `Get-Content extfile-client.cnf` 确认文件仅包含上述一行。
   openssl x509 -req -days 3650 -sha256 -in client.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out cert.pem -extfile extfile-client.cnf
   ```
   > 运行 `openssl genrsa -aes256 -out ca-key.pem 4096` 时，终端会提示 **Enter PEM pass phrase** 和 **Verifying - Enter PEM pass phrase**。这是在为 CA 私钥设置密码，请输入并牢记同一个口令；后续签发服务器/客户端证书时会再次要求输入。如果不希望设置密码，可将命令中的 `-aes256` 去掉。
3. 若需要去除服务器私钥密码：
   ```powershell
   openssl rsa -in server-key.pem -out server-key.pem
   ```

## 4. 配置 Docker Desktop

1. 将 `ca.pem`、`server-cert.pem`、`server-key.pem` 复制到 Docker Desktop 读取的配置目录。
   - **默认位置**：新版 Docker Desktop 会把 `daemon.json` 存在 `C:/ProgramData/DockerDesktop/config/`。少数旧版本或升级后的系统可能继续使用 `C:/ProgramData/Docker/config/`。
   - 先确定哪个目录存在（或需要创建）：
     ```powershell
     Test-Path C:\ProgramData\DockerDesktop
     Test-Path C:\ProgramData\Docker
     ```
     如果两个目录都存在，优先使用 `DockerDesktop` 目录；若只有 `Docker` 目录，则在其下创建 `config` 子目录。
   - 使用 `New-Item` 创建所需的 `config` 子目录：
     ```powershell
     # 任选其一，根据上一步的检测结果执行
     New-Item -ItemType Directory -Force -Path C:\ProgramData\DockerDesktop\config
     New-Item -ItemType Directory -Force -Path C:\ProgramData\Docker\config
     ```
     即使目录本身已经出现，也可以带 `-Force` 再执行一次，确保 `config` 子目录存在。
   - 然后从证书生成目录复制文件（请选择与你使用的目录相符的目标路径）：
     ```powershell
     # 将证书复制到 DockerDesktop 目录
     Copy-Item -Path C:\docker-certs\ca.pem,C:\docker-certs\server-cert.pem,C:\docker-certs\server-key.pem -Destination C:\ProgramData\DockerDesktop\config\

     # 如果你明确打算让 Docker 使用 C:\ProgramData\Docker\config\
     Copy-Item -Path C:\docker-certs\ca.pem,C:\docker-certs\server-cert.pem,C:\docker-certs\server-key.pem -Destination C:\ProgramData\Docker\config\
     ```
     只要与后续 `daemon.json` 中的证书路径保持一致即可，不需要同时复制到两个位置。
2. 打开 Docker Desktop → **Settings → Resources → Docker Engine**，在原有 JSON 基础上合并如下字段：

   ```json
   {
     "builder": {
       "gc": {
         "defaultKeepStorage": "20GB",
         "enabled": true
       }
     },
     "experimental": false,
     "hosts": [
       "tcp://0.0.0.0:2376",
       "npipe://"
     ],
     "tlsverify": true,
     "tlscacert": "C:/ProgramData/DockerDesktop/config/ca.pem",
     "tlscert": "C:/ProgramData/DockerDesktop/config/server-cert.pem",
     "tlskey": "C:/ProgramData/DockerDesktop/config/server-key.pem"
   }
   ```
   如果你选择让 Docker 使用 `C:/ProgramData/Docker/config/`，请将上面三行证书路径替换为 `C:/ProgramData/Docker/config/...`。
   > ⚠️ **不要直接修改** `C:/Program Files/Docker/Docker/resources/windows-daemon-options.json` 或 `linux-daemon-options.json`。
   > 这些文件只是 Docker Desktop 在首次启动时用来生成默认配置的模板，即便改动也不会被后续的 Apply & Restart 读取，反而可能在升级时被覆盖。
   > 正确做法是在 Docker Desktop 设置页面（或 `C:/ProgramData/DockerDesktop/config/daemon.json`）中编辑实际生效的配置。
   >
   > 如果 `C:/ProgramData/DockerDesktop/config/` 目录中尚未出现 `daemon.json`，无需担心——在此界面粘贴上述 JSON 并点击 **Apply & Restart** 后，Docker Desktop 会自动创建该文件。
   > 如果你希望在界面外先手动生成一个最小配置，也可以执行：
   > ```powershell
   > Set-Content -Encoding ascii -Path C:\ProgramData\DockerDesktop\config\daemon.json -Value '{"experimental":false}'
   > ```
   > 随后再次打开 **Settings → Resources → Docker Engine**，就能在界面中看到刚写入的 JSON，再继续补充 `hosts` 与 TLS 相关字段。
3. 点击 **Apply & Restart**。
   - 如果界面长时间卡住或直接弹出 **Reset** 提示，而且 `daemon.json` 依旧没有生成，通常是因为 Docker 在写入前检测到 JSON 无效或证书路径缺失。此时可以：
     1. 复制界面中的 JSON 到记事本（或 VS Code），确认没有中文标点、全角空格及 BOM；必要时先在管理员 PowerShell 中执行
        ```powershell
        Set-Content -Encoding ascii -Path C:\ProgramData\DockerDesktop\config\daemon.json -Value '{"experimental":false}'
        ```
        预先创建一个最小配置，再返回设置界面增量添加字段。
     2. 若之前手动编辑过 `daemon.json` 并导致 Docker 无法启动，可把旧文件重命名为 `daemon.json.bak`，然后重复上一步重新写入最小配置。
     3. 确认已以管理员身份运行 Docker Desktop，或者右键 “Docker Desktop” → **以管理员身份运行** 后再尝试 **Apply & Restart**。
   - 若仍然弹出“重置设置”，先不要立即恢复出厂设置，改为在管理员 PowerShell 中检查：
     ```powershell
     Test-Path C:\ProgramData\DockerDesktop\config\ca.pem
     Test-Path C:\ProgramData\DockerDesktop\config\server-cert.pem
     Test-Path C:\ProgramData\DockerDesktop\config\server-key.pem
     ```
     任何 `False` 结果都表示文件未放到预期目录。
   - 若仍无法启动，可用管理员权限运行 Docker Desktop 附带的守护进程调试命令查看报错：
     ```powershell
     & 'C:\Program Files\Docker\Docker\resources\dockerd.exe' -D --config-file "C:\ProgramData\DockerDesktop\config\daemon.json"
     ```
     > 如果你让 Docker 使用 `C:\ProgramData\Docker\config\daemon.json`，请将 `--config-file` 路径替换成对应位置。

     * `--config-file` 显式指定 Docker Desktop 正在使用的配置文件，避免单独运行 `dockerd.exe -D` 时只加载默认的命名管道设置（日志中只看到
       `API listen on //./pipe/docker_engine`）。
     * 如果终端输出是乱码，可先执行 `chcp 65001` 或在 PowerShell 中运行 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`，将控制台
       编码切换为 UTF-8 后再运行上述命令。
     * 根据调试输出提示修正路径、证书内容或权限后，再回到 Docker Desktop 中重试 **Apply & Restart**。只有当配置完全无法恢复时，才需要在
       *Settings → Troubleshoot* 中执行 *Reset to factory defaults*。

4. **确认配置已生效**。
   - 打开管理员 PowerShell，检查 Docker Desktop 保存下来的 `daemon.json` 内容：
     ```powershell
     Get-Content C:\ProgramData\DockerDesktop\config\daemon.json
     ```
     确认其中存在 `"hosts": ["tcp://0.0.0.0:2376", "npipe://"]` 与 TLS 证书字段。
   - 重新运行调试守护进程，并留意最后一行是否出现 `API listen on 0.0.0.0:2376`：
     ```powershell
     & 'C:\Program Files\Docker\Docker\resources\dockerd.exe' -D --config-file "C:\ProgramData\DockerDesktop\config\daemon.json"
     ```
    如输出为乱码，可先执行 `chcp 65001` 或 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` 后再运行上面的命令。如果输出仍旧只有
    `API listen on //./pipe/docker_engine`，请确认：
    1. `--config-file` 参数指向的路径与你实际保存 `daemon.json` 的位置一致。最简单的办法是在 PowerShell 中依次运行
       ```powershell
       Get-Content C:\ProgramData\DockerDesktop\config\daemon.json
       Get-Content C:\ProgramData\Docker\config\daemon.json
       ```
       与 Docker Desktop 设置界面中显示的 JSON 一致的那个文件，就是当前生效的配置。随后把该路径填入 `--config-file`，并确保命令
       行字符串使用与你复制证书时相同的目录。
    2. `daemon.json` 中确实包含 `"hosts": ["tcp://0.0.0.0:2376", "npipe://"]`。
    若路径正确仍未监听 2376 端口，请回到 Docker Desktop → **Settings → Resources → Docker Engine**，确认界面中的 JSON 与上一步
     `daemon.json` 内容一致后再次点击 **Apply & Restart**。
   - 也可以直接验证端口监听情况：
     ```powershell
     netstat -ano | findstr 2376
     ```
     若看到 `0.0.0.0:2376` 或本机 IP:2376 的 `LISTENING` 条目，则说明远程 API 已经开启。

5. **不要使用 Settings → Resources → Proxies 来开放远程 API。**
   - 该页面仅用于为 Docker 客户端访问外部网络（例如拉取镜像）配置 HTTP/HTTPS/SOCKS5 代理，无法把 Docker 守护进程暴露到公网。
   - 即使勾选 *Manual proxy configuration* 并填写端口，也只是让 Docker 请求通过代理服务器转发，不会在本机监听 2376 端口。
   - 若需要远程访问 Docker 守护进程，必须按照上面的步骤在 `daemon.json` 中显式添加 `"hosts": ["tcp://0.0.0.0:2376", "npipe://"]` 并启用 TLS 证书；代理设置不能替代这些配置。

## 5. 防火墙与端口开放

- 在 Windows Defender 防火墙中新建入站规则，仅允许可信 IP 访问 TCP 2376。
- 如在云服务器上部署，还需在云控制台安全组中开放同样端口。

## 6. Agent 主机连接测试

1. 将 `ca.pem`、`cert.pem`、`key.pem` 复制到 Agent 主机并设置权限。
2. 在 Agent 主机终端中设置环境变量：
   ```bash
   export DOCKER_HOST=tcp://<Windows公网IP>:2376
   export DOCKER_TLS_VERIFY=1
   export DOCKER_CERT_PATH=/path/to/certs
   ```
3. 运行 `docker info` 验证是否能看到 Windows 主机的 Docker 信息。

## 7. 准备 SWE-bench 镜像

- 在 Windows 主机执行：
  ```powershell
  docker pull ghcr.io/princeton-nlp/swe-bench:latest
  ```
- 若使用自定义镜像，请确保 `sandbox.docker_image` 字段与本地标签一致。

## 8. 运行 DeepSWE/RepoEnv

- 运行 DeepSWE 前确保已导出远程 Docker 相关环境变量。
- 将 `rollout_engine_args.base_url` 指向部署好的模型推理服务公网地址。
- 启动脚本后，RepoEnv 会通过远程 Docker API 在 Windows 主机上创建 SWE-bench 容器，Agent 输出的命令会在该宿主机执行。

通过以上步骤，即可在 Windows PC 上搭建同时提供推理与 RepoEnv 执行的远程 Docker 环境，并解决 OpenSSL 相关报错。
