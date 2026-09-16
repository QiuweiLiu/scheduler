# 公开仓库审阅（GPT，2026-09-17）

仓库：https://github.com/QiuweiLiu/scheduler （public，901 文件，113.6 MB）
审阅范围：T1/T2/T3 统计实现、消费函数与调度器、实验运行器/可复现性。

---

整体判断：仓库已经从“实验代码”升级到可审计研究仓库，但还不建议投稿冻结。统计主结论基本可信；r95 正交对照也已经比早期版本干净。当前最重要的是两个实现级 P0：CP-RHO 量纲和实验 provenance/resume。

1. T1/T2/T3：6 项修复均已真正落地。 MixedLM 已显式 re_formula="1"+REML；T1 的 MOM 确实在固定效应残差上计算并同时输出 raw/clamped 分量；直接 within-run covariance 保留负值；video 解析 fail-closed；cluster bootstrap 同时重命名 video/run；T3 已改为 (type,stack,baseline) 内经验 PIT + pair-specific marginal lift。

但仍有三点需修。第一，当前 harmonic-mean MOM 对不平衡 nested design 只是近似矩估计，建议继续作为主描述量，但论文不要称 unbiased variance-component estimator；直接 covariance 是很好的符号诊断。第二，代码实际把 T1 bootstrap 截成最多 400 次，T2/T3 默认仅 200、permutation 300，投稿版应统一 ≥2000。第三，目前 permutation p-value 是 P(null >= observed)，只检验“正 lift”；你们观测 lift<1 时它不能证明负尾依赖，应同时输出 lower-tail/two-sided p。另建议 T1 增加 type×stack/type×baseline rich-fixed-effect sensitivity。

2. 消费函数。 r50/r90/r95/r50k 现在是很干净的 runtime-only 正交族：current 都用同一 table-p50，future load 排除；k=6.2293 也明确冻结于 dev，不是事后 confirm 调参。 场景族的旧问题大体修掉了：cache key 加 context、完整槽位、N=128、凸组合 (1−κ)E+κCVaR。

这里仍有两个公平性边界：scen128/comon128 用 lognormal 重建，sigma 主要按 p95/p50 拟合，并非严格满足三处分位数；所以论文应写“under our lognormal reconstruction”，不能泛称所有 CVaR。其次 comonotone 共享 runtime/load 的 U，但length 仍独立抽样，严格说是 conditional-comonotone，不是整条链完全共单调。建议补一个固定 argmax-L 的 comon/scenario 对照。

更严重的是 CP-RHO：end_cost = 1000×weight×end_ms，但 future_cost = weight×future_ms，所以 future 项相对 completion 项被缩小约 1000 倍；而 fallback 又把 current/future 都除以 100000，二者语义不一致。这个基线修复后必须重跑。

tail_shuffle 的结果报告存在，但生成脚本写成 .scratch/phase11_stats.py，公开仓库里没有；因此目前第三方无法从原 artifacts 重建该关键对照。

3. 可复现性。 README 对“纯仿真、固定实测 runtime、哪些数据未公开”说明得不错。 但 configs/requirements.txt 实际不存在；README 却给了这条安装命令。更关键的是 runner 的 resume 只按 (episode_id, policy) 跳过，没有验证代码/input/artifact fingerprint；换过代码或 artifact 后复用同一 output-dir，会把旧结果静默混入新报告。 另外 normalize_full5_artifacts.py 修改 gzip 内容后直接复制旧 b05_artifact_manifest.json，其文件哈希因此失效。

投稿前修复清单：

P0：修 CP-RHO 1000×单位；resume 加 git SHA+输入/代码/artifact hash fingerprint 并 mismatch fail；normalizer 重算 manifest；公开 tail-shuffle 生成脚本；提供真实 requirements.lock/environment.yml。

P1：T1/T3 bootstrap/permutation ≥2000+双侧检验；补 rich-fixed T1；补 fixed-L scenario/comon；公开 confirm300 ID 与 paired-bootstrap 脚本。

P2：添加 LICENSE、seed/provenance 总表、one-command reproduce_main.sh；README 把“可从 processed data 再生”改成“主仿真可复现，但原视频/原 trace/权重不可再生”。

如果这几个 P0 关闭，我会认为仓库已经达到论文 artifact 可提交的质量。
