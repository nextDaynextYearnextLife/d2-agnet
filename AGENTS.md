## 项目概述
Dota 2 比赛数据分析与胜率预测项目（d2-agnet）。通过机器学习模型预测 Radiant 方的胜率。

## 技术栈
- Python 3
- scikit-learn（随机森林分类器）
- pandas（数据处理）
- HTML/CSS（可视化报告）

## 目录结构
```
d2-agnet/
├── build_model.py           # 训练随机森林模型
├── feature_engineering.py   # 特征工程
├── preprocess_data.py       # 数据预处理
├── get_data.py              # 数据获取
├── download_hero_images.py  # 下载英雄图片
├── generate_report.py       # 生成可视化报告
├── matches.csv              # 原始比赛数据
├── processed_matches.csv    # 处理后的数据
├── random_forest_model.pkl  # 训练好的模型
└── visualization/
    └── report.html          # 可视化报告
```

## 关键入口 / 核心模块
- `build_model.py` - 主训练脚本，使用 RandomForestClassifier 预测胜率
- `generate_report.py` - 生成 HTML 可视化报告

## 运行方式
```bash
python build_model.py    # 训练模型
python generate_report.py  # 生成报告
```

## 用户偏好与长期约束
- 项目为纯离线数据处理，无需网络服务
- 模型使用 class_weight='balanced' 处理类别不平衡

## 常见问题和预防
- 确保 processed_matches.csv 存在再运行 build_model.py
- 模型精度受限于数据集规模和特征质量
