# Git 分支策略

## 分支说明

### `main` 分支（主分支）✅
- **用途**: 主开发分支，所有开发工作都在此进行
- **状态**: 活跃开发中
- **说明**: 所有提交者信息已修正为 `jengzang`
- **推荐**: 这是唯一的主分支

## 工作流程

### 日常开发
```bash
# 确保在 main 分支
git checkout main

# 拉取最新代码
git pull origin main

# 进行开发工作
# ... 修改代码 ...

# 提交更改
git add .
git commit -m "描述你的更改"

# 推送到远程
git push origin main
```

### 创建功能分支（可选）
```bash
# 从 main 创建功能分支
git checkout -b feature/new-feature main

# 开发完成后合并回 main
git checkout main
git merge feature/new-feature

# 删除功能分支
git branch -d feature/new-feature
```

## Git 配置

确保你的 git 配置正确：

```bash
# 检查配置
git config user.name
git config user.email

# 应该显示：
# jengzang
# jengzang@users.noreply.github.com
```

如果不正确，运行：
```bash
git config user.name "jengzang"
git config user.email "jengzang@users.noreply.github.com"
```

## 分支历史

项目已简化为单分支模式：

| 日期 | 变更 | 说明 |
|------|------|------|
| 2026-03-02 | 创建 `main` 分支 | 从 `dev` 改名而来 |
| 2026-03-02 | 删除 `master` 和 `dev` | 简化为单分支 |
| 2026-03-02 | 修正所有提交者信息 | 统一为 `jengzang` |

## 为什么使用 `main`？

- ✅ **现代标准**: GitHub 推荐使用 `main` 作为默认分支名
- ✅ **简单明了**: 只有一个主分支，避免混淆
- ✅ **易于维护**: 不需要在多个分支间同步
- ✅ **包容性**: 更中性的命名

## 注意事项

1. **所有开发都在 main 分支进行**
2. **定期推送到远程仓库**
3. **提交信息要清晰明确**
4. **可以创建临时功能分支，但最终都合并回 main**

## 克隆仓库

```bash
# 克隆仓库（会自动 checkout 到 main 分支）
git clone https://github.com/jengzang/dialect-ssl-tts-pipeline.git

# 进入项目目录
cd dialect-ssl-tts-pipeline

# 查看当前分支
git branch
# 应该显示: * main
```

---

**最后更新**: 2026-03-02
**维护者**: jengzang (不羁)
