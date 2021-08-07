# A Neural Framework for Chinese Medical Named Entity Recognition

> **Author: [StevenChaoo](https://github.com/StevenChaoo)**

![vscode](https://img.shields.io/badge/visual_studio_code-007acc?style=flat-square&logo=visual-studio-code&logoColor=ffffff)![neovim](https://img.shields.io/badge/Neovim-57a143?style=flat-square&logo=Neovim&logoColor=ffffff)![git](https://img.shields.io/badge/Git-f05032?style=flat-square&logo=git&logoColor=ffffff)![python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=Python&logoColor=ffffff)

This blog is written by **Neovim** and **Visual Studio Code**. You may need to clone this repository to your local and use **Visual Studio Code** to read. ***Markdown Preview Enhanced*** plugin is necessary as well. Codes are all writen with **Python**.

This repository contains (PyTorch) code and pre-trained models for Biomedical Named Entity Recognition task, described by the paper: [*A Neural Framework for Chinese Medical Named Entity Recognition*](https://link.springer.com/chapter/10.1007/978-3-030-59605-7_6). This code is also the implementation of my B.S. thesis: *基于神经网络的中文电子病历命名实体识别*.

## Quick links

- [A Neural Framework for Chinese Medical Named Entity Recognition](#a-neural-framework-for-chinese-medical-named-entity-recognition)
  - [Quick links](#quick-links)
  - [Setup](#setup)
    - [Install dependencies](#install-dependencies)
    - [Data preprocessing](#data-preprocessing)
  - [Run CRF model](#run-crf-model)
  - [Run BiLSTM-RNN model](#run-bilstm-rnn-model)
  - [Run BERT-based model](#run-bert-based-model)
    - [Train & Fine-tune](#train--fine-tune)
    - [Evaluate](#evaluate)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Setup

### Install dependencies

Please install all the dependency packages using the following command:

```bash
pip install -r requirements.txt
```

### Data preprocessing

Our experiments are based on the CCKS 2019 task 1 dataset which contains 1k real Chinese biomedical electronic records.

```json
[
    {"originalText": "，患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗，患者恢复好，切口愈合良好。，术后病理示：直肠腺癌（中低度分化），浸润溃疡型，面积3.5*2CM，侵达外膜。双端切线另送“近端”、“远端”及环周底部切除面未查见癌。肠壁一站（10个）、中间组（8个）淋巴结未查见癌。，免疫组化染色示：ERCC1弥漫（+）、TS少部分弱（+）、SYN（-）、CGA（-）。术后查无化疗禁忌后给予3周期化疗，，方案为：奥沙利铂150MG D1，亚叶酸钙0.3G+替加氟1.0G D2-D6，同时给与升白细胞、护肝、止吐、免疫增强治疗，患者副反应轻。院外期间患者一般情况好，无恶心，无腹痛腹胀胀不适，无现患者为行复查及化疗再次来院就诊，门诊以“直肠癌术后”收入院。   近期患者精神可，饮食可，大便正常，小便正常，近期体重无明显变化。", "entities": [{"label_type": "疾病和诊断", "overlap": 0, "start_pos": 8, "end_pos": 11}, {"label_type": "手术", "overlap": 0, "start_pos": 21, "end_pos": 35}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 78, "end_pos": 95}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 139, "end_pos": 159}, {"end_pos": 234, "label_type": "药物", "overlap": 0, "start_pos": 230}, {"end_pos": 247, "label_type": "药物", "overlap": 0, "start_pos": 243}, {"end_pos": 255, "label_type": "药物", "overlap": 0, "start_pos": 252}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 276, "end_pos": 277}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 312, "end_pos": 313}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 314, "end_pos": 315}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 342, "end_pos": 347}]},
    {"originalText": "，患者因罹患“胃癌”于2013-10-29在我院予行全麻上胃癌根治术，，术中见：腹腔内腹水，腹膜无转移，肝脏未触及明显转移性灶，肿瘤位于胃体、胃底部，小弯侧偏后壁，约5*4*2CM大小，肿瘤已侵达浆膜外，第1、3组淋巴结肿大，肿瘤尚能活动，经探查决定行全胃切除，空肠J字代胃术。手术顺利，术后积极予相关对症支持治疗；，后病理示：胃底、体小弯侧低分化腺癌，部分为印戒细胞癌图像，蕈伞型，面积5.2*3.5CM，局部侵达粘膜上层，并于少数腺管内查见癌栓。双端切线及另送“近端切线”未查见癌。呈三组（5/13个）淋巴结癌转移。一组（7个）、四组（13个）、五组（1个）、六组（4个）淋巴结未查见癌。，癌组织免疫组化染色示：ERCC1（+）、β-TUBULIN-III（+）、TS（-）、RRM1（-）、TOPOII阳性细胞数约20%、CERBB-2（2+） 。依据患者病情及肿瘤病理与分期继续术后辅助性化疗指征存在，患者及家属拒绝化疗。自术后出院以来，患者一般情况保持良好；无发热，偶有恶心，无呕吐，无反酸、嗳气，无明显进食不适，偶有进食后轻微腹胀，无腹痛。现患者为行进一步复查并必要时适当处理而再来我院就诊，门诊依情以“胃恶性肿瘤术后”收入院。目前患者精神及情绪状态良好，食欲较术前明显减少，饮食可，夜间睡眠后；今8个月体重减轻18KG。", "entities": [{"label_type": "疾病和诊断", "overlap": 0, "start_pos": 7, "end_pos": 9}, {"end_pos": 34, "label_type": "手术", "overlap": 0, "start_pos": 29}, {"end_pos": 42, "label_type": "解剖部位", "overlap": 0, "start_pos": 40}, {"end_pos": 44, "label_type": "解剖部位", "overlap": 0, "start_pos": 43}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 46, "end_pos": 47}, {"end_pos": 54, "label_type": "解剖部位", "overlap": 0, "start_pos": 52}, {"end_pos": 70, "label_type": "解剖部位", "overlap": 0, "start_pos": 68}, {"end_pos": 74, "label_type": "解剖部位", "overlap": 0, "start_pos": 71}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 75, "end_pos": 78}, {"end_pos": 138, "label_type": "手术", "overlap": 0, "start_pos": 126}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 164, "end_pos": 191}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 244, "end_pos": 256}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 260, "end_pos": 291}, {"end_pos": 470, "label_type": "解剖部位", "overlap": 0, "start_pos": 469}, {"end_pos": 474, "label_type": "解剖部位", "overlap": 0, "start_pos": 473}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 508, "end_pos": 515}]},
    {"originalText": "，患者3月余前于我院诊断为“直肠癌”，于2015-10-26在全麻上行腹腔镜直肠癌根治术，，术后病理示：，201518502：（直肠）腺癌（中度分化），浸润溃疡型，体积2.7*2*0.8CM，侵达浆膜。 双端切线及另送“近切线”、“远切线”未查见癌。 肠壁一站（6个）、中间组（3个）、中央组（3个）淋巴结未查见癌。低级别腺管状腺瘤。，免疫组化染色示：TS部分（+）、SYN（-）。，术后病理分期：PT3N0M0，II期，DUKES B。依情2015-11-08.2015-12-09给予奥沙利铂200MG D1+亚叶酸钙0.3G D2-6 +替加氟1G D2-6 静滴，同时辅以镇吐、升血、免疫调节等对症支持治疗。化疗过程总体顺利。现为复查化疗来我院，门诊以“直肠癌术后”收入院。目前患者精神好，食欲及饮食好，夜间睡眠良好，小便正常，大便4-5次/天，基本成形。否认近期明显体重变化。", "entities": [{"label_type": "疾病和诊断", "overlap": 0, "start_pos": 14, "end_pos": 17}, {"end_pos": 44, "label_type": "手术", "overlap": 0, "start_pos": 35}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 63, "end_pos": 81}, {"label_type": "解剖部位", "overlap": 0, "start_pos": 126, "end_pos": 153}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 158, "end_pos": 166}, {"end_pos": 248, "label_type": "药物", "overlap": 0, "start_pos": 244}, {"end_pos": 261, "label_type": "药物", "overlap": 0, "start_pos": 257}, {"end_pos": 275, "label_type": "药物", "overlap": 0, "start_pos": 272}, {"label_type": "疾病和诊断", "overlap": 0, "start_pos": 331, "end_pos": 336}]},
    ...
]
```

Firstly, we sorted the data into two formats, both of which split the training set, verification set and test set according to 7:1:2. In the first type, each row consists of three columns, namely **token**, **space**, and **label corresponding to the token**. Separate different sentences with blank lines.

```txt
， O
患 O
者 O
3 O
月 O
前 O
因 O
“ O
直 B-DIS
肠 M-DIS
癌 E-DIS
” O
...
```

The second type stores sentences and labels in `sentences.txt` and `tags.txt`, respectively, and writes all label types in `tags.txt` in the upper directory.

```txt
# sentences.txt
， 患 者 3 月 前 因 “ 直 肠 癌 ” 于 在 我 院 于 全 麻 上 行 直 肠 癌 根 治 术 （ D I X O N 术 ） ， 手 术 过 程 顺 利 ， 术 后 给 予 抗 感 染 及 营 养 支 持 治 疗 ， 患 者 恢 复 好 ， 切 口 愈 合 良 好 。 ， 术 后 病 理 示 ： 直 肠 腺 癌 （ 中 低 度 分 化 ） ， 浸 润 溃 疡 型 ， 面 积 3 . 5 * 2 C M ， 侵 达 外 膜 。 双 端 切 线 另 送 “ 近 端 ” 、 “ 远 端 ” 及 环 周 底 部 切 除 面 未 查 见 癌 。 肠 壁 一 站 （ 1 0 个 ） 、 中 间 组 （ 8 个 ） 淋 巴 结 未 查 见 癌 。 ， 免 疫 组 化 染 色 示 ： E R C C 1 弥 漫 （ + ） 、 T S 少 部 分 弱 （ + ） 、 S Y N （ - ） 、 C G A （ - ） 。 术 后 查 无 化 疗 禁 忌 后 给 予 3 周 期 化 疗 ， ， 方 案 为 ： 奥 沙 利 铂 1 5 0 M G <SPACE> D 1 ， 亚 叶 酸 钙 0 . 3 G + 替 加 氟 1 . 0 G <SPACE> D 2 - D 6 ， 同 时 给 与 升 白 细 胞 、 护 肝 、 止 吐 、 免 疫 增 强 治 疗 ， 患 者 副 反 应 轻 。 院 外 期 间 患 者 一 般 情 况 好 ， 无 恶 心 ， 无 腹 痛 腹 胀 胀 不 适 ， 无 现 患 者 为 行 复 查 及 化 疗 再 次 来 院 就 诊 ， 门 诊 以 “ 直 肠 癌 术 后 ” 收 入 院 。 <SPACE> <SPACE> <SPACE> 近 期 患 者 精 神 可 ， 饮 食 可 ， 大 便 正 常 ， 小 便 正 常 ， 近 期 体 重 无 明 显 变 化 。 
， 患 者 因 罹 患 “ 胃 癌 ” 于 2 0 1 3 - 1 0 - 2 9 在 我 院 予 行 全 麻 上 胃 癌 根 治 术 ， ， 术 中 见 ： 腹 腔 内 腹 水 ， 腹 膜 无 转 移 ， 肝 脏 未 触 及 明 显 转 移 性 灶 ， 肿 瘤 位 于 胃 体 、 胃 底 部 ， 小 弯 侧 偏 后 壁 ， 约 5 * 4 * 2 C M 大 小 ， 肿 瘤 已 侵 达 浆 膜 外 ， 第 1 、 3 组 淋 巴 结 肿 大 ， 肿 瘤 尚 能 活 动 ， 经 探 查 决 定 行 全 胃 切 除 ， 空 肠 J 字 代 胃 术 。 手 术 顺 利 ， 术 后 积 极 予 相 关 对 症 支 持 治 疗 ； ， 后 病 理 示 ： 胃 底 、 体 小 弯 侧 低 分 化 腺 癌 ， 部 分 为 印 戒 细 胞 癌 图 像 ， 蕈 伞 型 ， 面 积 5 . 2 * 3 . 5 C M ， 局 部 侵 达 粘 膜 上 层 ， 并 于 少 数 腺 管 内 查 见 癌 栓 。 双 端 切 线 及 另 送 “ 近 端 切 线 ” 未 查 见 癌 。 呈 三 组 （ 5 / 1 3 个 ） 淋 巴 结 癌 转 移 。 一 组 （ 7 个 ） 、 四 组 （ 1 3 个 ） 、 五 组 （ 1 个 ） 、 六 组 （ 4 个 ） 淋 巴 结 未 查 见 癌 。 ， 癌 组 织 免 疫 组 化 染 色 示 ： E R C C 1 （ + ） 、 β - T U B U L I N - I I I （ + ） 、 T S （ - ） 、 R R M 1 （ - ） 、 T O P O I I 阳 性 细 胞 数 约 2 0 % 、 C E R B B - 2 （ 2 + ） <SPACE> 。 依 据 患 者 病 情 及 肿 瘤 病 理 与 分 期 继 续 术 后 辅 助 性 化 疗 指 征 存 在 ， 患 者 及 家 属 拒 绝 化 疗 。 自 术 后 出 院 以 来 ， 患 者 一 般 情 况 保 持 良 好 ； 无 发 热 ， 偶 有 恶 心 ， 无 呕 吐 ， 无 反 酸 、 嗳 气 ， 无 明 显 进 食 不 适 ， 偶 有 进 食 后 轻 微 腹 胀 ， 无 腹 痛 。 现 患 者 为 行 进 一 步 复 查 并 必 要 时 适 当 处 理 而 再 来 我 院 就 诊 ， 门 诊 依 情 以 “ 胃 恶 性 肿 瘤 术 后 ” 收 入 院 。 目 前 患 者 精 神 及 情 绪 状 态 良 好 ， 食 欲 较 术 前 明 显 减 少 ， 饮 食 可 ， 夜 间 睡 眠 后 ； 今 8 个 月 体 重 减 轻 1 8 K G 。 
， 患 者 3 月 余 前 于 我 院 诊 断 为 “ 直 肠 癌 ” ， 于 2 0 1 5 - 1 0 - 2 6 在 全 麻 上 行 腹 腔 镜 直 肠 癌 根 治 术 ， ， 术 后 病 理 示 ： ， 2 0 1 5 1 8 5 0 2 ： （ 直 肠 ） 腺 癌 （ 中 度 分 化 ） ， 浸 润 溃 疡 型 ， 体 积 2 . 7 * 2 * 0 . 8 C M ， 侵 达 浆 膜 。 <SPACE> 双 端 切 线 及 另 送 “ 近 切 线 ” 、 “ 远 切 线 ” 未 查 见 癌 。 <SPACE> 肠 壁 一 站 （ 6 个 ） 、 中 间 组 （ 3 个 ） 、 中 央 组 （ 3 个 ） 淋 巴 结 未 查 见 癌 。 低 级 别 腺 管 状 腺 瘤 。 ， 免 疫 组 化 染 色 示 ： T S 部 分 （ + ） 、 S Y N （ - ） 。 ， 术 后 病 理 分 期 ： P T 3 N 0 M 0 ， I I 期 ， D U K E S <SPACE> B 。 依 情 2 0 1 5 - 1 1 - 0 8 . 2 0 1 5 - 1 2 - 0 9 给 予 奥 沙 利 铂 2 0 0 M G <SPACE> D 1 + 亚 叶 酸 钙 0 . 3 G <SPACE> D 2 - 6 <SPACE> + 替 加 氟 1 G <SPACE> D 2 - 6 <SPACE> 静 滴 ， 同 时 辅 以 镇 吐 、 升 血 、 免 疫 调 节 等 对 症 支 持 治 疗 。 化 疗 过 程 总 体 顺 利 。 现 为 复 查 化 疗 来 我 院 ， 门 诊 以 “ 直 肠 癌 术 后 ” 收 入 院 。 目 前 患 者 精 神 好 ， 食 欲 及 饮 食 好 ， 夜 间 睡 眠 良 好 ， 小 便 正 常 ， 大 便 4 - 5 次 / 天 ， 基 本 成 形 。 否 认 近 期 明 显 体 重 变 化 。 

# tags.txt
O O O O O O O O B-DIS M-DIS M-DIS O O O O O O O O O O B-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-MED M-MED M-MED M-MED O O O O O O O O O B-MED M-MED M-MED M-MED O O O O O B-MED M-MED M-MED O O O O O O O O O O O O O O O O O O O O O B-ANA O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-ANA O B-ANA O O O O O O O O O O O O O O O O O O O O O O O O O O O B-DIS M-DIS M-DIS M-DIS M-DIS O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O 
O O O O O O O B-DIS M-DIS O O O O O O O O O O O O O O O O O O O O B-OPE M-OPE M-OPE M-OPE M-OPE O O O O O O B-ANA M-ANA O B-ANA O O B-ANA O O O O O B-ANA M-ANA O O O O O O O O O O O O O O B-ANA M-ANA O B-ANA M-ANA M-ANA O B-ANA M-ANA M-ANA O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE O O O O O O O O O O O O O O O O O O O O O O O O O O B-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA O O O O B-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-ANA O O O B-ANA O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O 
O O O O O O O O O O O O O O B-DIS M-DIS M-DIS O O O O O O O O O O O O O O O O O O B-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE M-OPE O O O O O O O O O O O O O O O O O O O B-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA M-ANA O O O O O B-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS M-DIS O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-MED M-MED M-MED M-MED O O O O O O O O O B-MED M-MED M-MED M-MED O O O O O O O O O O O B-MED M-MED M-MED O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-DIS M-DIS M-DIS M-DIS M-DIS O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O 

# ../tags.txt
B-ANA
B-DIS
B-OPE
B-IMA
B-MED
B-LAB
M-ANA
M-DIS
M-OPE
M-IMA
M-MED
M-LAB
O
```

## Run CRF model

```bash
python Core/run_crf.py \
    --data_dir Datasets/Raw_data.json \
    --encoding_type utf-8-sig \
    --seed 0
```

## Run BiLSTM-RNN model

```bash
python Core/run_rnn.py \
    --generate_data {}.(True if need to generate data, recommend when first run this code or False) \
    --raw_data_dir Datasets/Raw_data.json \ 
    --encoding_type utf-8-sig \
    --save_dir Datasets/ \
    --seed 0
```

## Run BERT-based model

### Train & Fine-tune

1. Download [model weight](https://drive.google.com/file/d/1wXdhFf4BiXcafiZw7psURNcNqy8rwFw4/view?usp=sharing) from Google Drive and unzip it into `Model/modelConfig`.
2. Preprocess data based on [Data preprocession](#data-preprocessing).
3. Run `Core/train_bert.py`

```bash
python Core/train_bert.py \
    --data_dir Datasets/data/ \
    --pretrain_model bert-base-chinese \
    --tagger_model_dur Model/modelConfig/ \
    --restore_dir Model/modelConfig
```

### Evaluate

```bash
python Core/eval_bert.py \
    --data_dir Datasets/data/ \
    --pretrain_model bert-base-chinese
```

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Zhengyi Zhao (zyzhao@uir.edu.cn / steven_z_@outlook.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

```bibtex
@inproceedings{zhao2020neural,
  title={A Neural Framework for Chinese Medical Named Entity Recognition},
  author={Zhao, Zhengyi and Zhou, Ziya and Xing, Weichuan and Wu, Junlin and Chang, Yuan and Li, Binyang},
  booktitle={International Conference on AI and Mobile Services},
  pages={74--83},
  year={2020},
  organization={Springer}
}
```
