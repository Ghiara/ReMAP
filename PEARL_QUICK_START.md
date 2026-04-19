# 🚀 PEARL Cheetah Baseline 快速开始

## 三步开始训练

### 1️⃣ 快速测试（5-10 分钟）

```bash
cd /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization
conda activate wangbo-cu113

# 运行快速配置测试（50次迭代，减少任务数）
python run_pearl_cheetah_baseline.py \
    --config configs/pearl_cheetah_multi_config_quick.json \
    --gpu 0 \
    --debug
```

**预期输出**：
- 看到训练进度条
- 输出目录：`output/debug/cheetah-multi-task_xxxx/`
- Log file: `logs/pearl_*.log`

---

### 2️⃣ 正式训练（12-24 小时）

```bash
# 使用完整配置（500次迭代，80个训练任务）
python run_pearl_cheetah_baseline.py \
    --config configs/pearl_cheetah_multi_config.json \
    --gpu 0 \
    --output-dir output/pearl_baseline_full
```

**或使用后台运行**（推荐）：

```bash
# 后台运行 + 日志输出
nohup python run_pearl_cheetah_baseline.py \
    --config configs/pearl_cheetah_multi_config.json \
    --gpu 0 \
    --output-dir output/pearl_baseline_full > logs/pearl_baseline.log 2>&1 &

# 查看进度
tail -f logs/pearl_baseline.log
```

---

### 3️⃣ 监控和分析结果

```bash
# 实时查看日志
tail -f logs/pearl_baseline.log

# 查看生成的结果
ls -lh output/pearl_baseline_full/cheetah-multi-task_*/

# 检查训练进度 CSV
head -20 output/pearl_baseline_full/cheetah-multi-task_*/progress.csv
```

---

## 📊 两个配置的对比

| 方面 | 快速版 | 完整版 |
|------|--------|--------|
| 训练任务数 | 20 | 80 |
| 评估任务数 | 10 | 40 |
| 迭代数 | 50 | 500 |
| **预计时间** | **10-15分钟** | **12-24小时** |
| 网络大小 | 200 | 300 |
| 批大小 | 128 | 256 |
| 用途 | 调试/验证环境 | baseline对比 |

---

## 📁 配置文件位置

已创建/修改的文件：

1. **运行脚本**  
   - `run_pearl_cheetah_baseline.py` - 主训练脚本
   - `train_pearl_cheetah_baseline.sh` - Shell启动脚本

2. **配置文件**  
   - `configs/pearl_cheetah_multi_config.json` - 完整配置（原有）
   - `configs/pearl_cheetah_multi_config_quick.json` - 快速测试配置（新建）

3. **文档**  
   - `PEARL_CHEETAH_GUIDE.md` - 完整指南
   - `PEARL_QUICK_START.md` - 本文件

---

## 🔧 自定义训练

### 修改网络大小
编辑配置文件中的 `net_size`：
```json
"net_size": 400  // 增大网络容量
```

### 修改学习率
```json
"policy_lr": 0.0001,    // 更小的学习率
"context_lr": 0.0001
```

### 修改任务数量
```json
"n_train_tasks": 100,   // 更多训练任务
"n_eval_tasks": 50
```

---

## ⚡ 性能优化建议

1. **GPU内存溢出**？
   - 减少 `batch_size` 和 `meta_batch`
   - 减少 `net_size`

2. **训练太快完成**？
   - 增加 `num_iterations`
   - 增加 `num_steps_per_eval`

3. **想要更好的结果**？
   - 增加 `n_train_tasks` 到 100+
   - 增加 `num_iterations` 到 1000
   - 调整 `kl_lambda` 来平衡编码器正则化

---

## 🎯 验收标准

你的baseline应该：
- ✅ 成功训练完成（不出错）
- ✅ 生成 `progress.csv` 显示性能上升
- ✅ 保存所有模型检查点
- ✅ 学习曲线平滑上升

---

## 常见问题速查

| 问题 | 解决方案 |
|------|--------|
| `ModuleNotFoundError: No module named 'rlkit'` | 检查conda环境，确保已安装项目依赖 |
| CUDA OOM | 减少 `batch_size`, `meta_batch`, 或 `net_size` |
| 训练太慢 | 增加 GPU 利用率检查：`nvidia-smi` |
| 日志找不到 | 检查 `logs/` 和 `output/` 目录是否创建 |

---

## 下一步

1. **快速验证**：运行快速配置确保环境正确
2. **提交完整训练**：启动完整baseline
3. **保存结果**：记录最佳performance指标
4. **对比分析**：用你的分层方法与之对比

祝训练顺利！🎉

