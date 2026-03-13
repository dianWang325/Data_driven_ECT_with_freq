# Git 维护备忘录

## 解决 443 连接问题
git config --global http.proxy http://127.0.0.1:7890

## 排除大文件 (.gitignore)
- 权重文件 (*.pth)
- 结果文件夹 (基于...数据权重与结果/)

## 推送命令
git add .
git commit -m "update logs"
git push origin main