# Git 分支策略

## 分支说明

### `master` 分支
- **用途**: 稳定版本，保留原始提交历史
- **状态**: 只读，不再直接提交
- **说明**: 包含项目的原始开发历史

### `dev` 分支（主开发分支）✅
- **用途**: 日常开发和新功能实现
- **状态**: 活跃开发中
- **说明**: 所有提交者信息已修正为 `jengzang`
- **推荐**: 所有新的开发工作都在此分支进行

## 工作流程

### 日常开发
```bash
# 确保在 dev 分支
git checkout dev

# 拉取最新代码
git pull origin dev

# 进行开发工作
# ... 修改代码 ...

# 提交更改
git add .
git commit -m "描述你的更改"

# 推送到远程
git push origin dev
```

### 创建功能分支（可选）
```bash
# 从 dev 创建功能分支
git checkout -b feature/new-feature dev

# 开发完成后合并回 dev
git checkout dev
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

## 分支对比

| 特性 | master | dev |
|------|--------|-----|
| 提交者 | JNU Dialect Speech Team | jengzang ✅ |
| 状态 | 归档 | 活跃开发 |
| 用途 | 历史记录 | 日常开发 |
| 推荐使用 | ❌ | ✅ |

## 注意事项

1. **不要直接在 master 分支开发**
2. **所有新代码都提交到 dev 分支**
3. **定期推送到远程仓库**
4. **提交信息要清晰明确**

---

**最后更新**: 2026-03-02
**维护者**: jengzang (不羁)
