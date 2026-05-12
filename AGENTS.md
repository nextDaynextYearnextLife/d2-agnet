## 项目概述
Dota 2 比赛数据分析与胜率预测项目（d2-agnet）。通过机器学习模型分析英雄选取对胜率的影响。

## 技术栈
- Python 3
- scikit-learn（随机森林分类器）
- pandas（数据处理）
- HTML/CSS（可视化报告）

## 目录结构
```
d2-agnet/
├── build_model.py           # 训练随机森林模型
├── feature_engineering.py   # 特征工程（生成英雄胜率特征）
├── preprocess_data.py       # 数据预处理
├── get_data.py              # 从 OpenDota API 获取比赛数据
├── download_hero_images.py  # 下载英雄图片
├── generate_report.py       # 生成可视化报告
├── matches.csv              # 原始比赛数据（10000+条）
├── cleaned_matches.csv      # 清洗后的数据
├── processed_matches.csv   # 特征工程后的数据
├── hero_statistics.csv      # 英雄胜率统计
├── random_forest_model.pkl  # 训练好的模型
└── visualization/
    ├── hero_images/         # 英雄图片
    └── report.html          # 可视化报告
```

## 关键入口 / 核心模块
- `get_data.py` - 从 OpenDota API 获取有效比赛数据（过滤无英雄数据的比赛）
- `feature_engineering.py` - 为每个英雄创建出现特征 + 统计胜率
- `build_model.py` - 训练随机森林模型，分析特征重要性
- `generate_report.py` - 生成 HTML 可视化报告

## 运行方式
```bash
python get_data.py              # 获取新数据
python preprocess_data.py       # 清洗数据
python feature_engineering.py   # 特征工程
python build_model.py           # 训练模型
python generate_report.py       # 生成报告
```

## 数据说明
- 目标变量：`radiant_win`（Radiant 方是否获胜）
- 英雄特征：`hero_X_radiant`（英雄 X 是否在 Radiant 方）
- 胜率统计：基于英雄被选取时的 Radiant 方胜率

## 依赖安装
```bash
pip install scikit-learn pandas requests Pillow
```

## 用户偏好与长期约束
- 项目为纯离线数据处理，无需网络服务
- 模型使用 class_weight='balanced' 处理类别不平衡
- 英雄图片优先使用本地缓存，CDN 为备用源

## 常见问题和预防
- OpenDota publicMatches API 返回的数据需过滤无效记录（radiant_team = [0,0,0,0,0]）
- 模型准确率受限于数据集规模和特征质量（约 54-60%）
- 英雄胜率统计基于样本量，最少需要 10+ picks 才可靠
