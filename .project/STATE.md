## 指标口径审计（2026-09-18）— 前提撤回

- **"预测器严重失准"不成立**：用同一套统计代码给 J 验证集打分，**逐位复现**验收数字
  （覆盖 0.5555/0.9166/0.9680、RuntimeQScore 845.0）⇒ 两套打分同尺。
- **域内 Σp50/Σ真值 = 0.6322（不是 1.0）**；这是"逐步中位数之和 ≠ 总和的中位数"的必然结果
  （真值中位 2209 ms vs 均值 3877 ms，1.76×）。
- **主指标 RuntimeQScore：R7 = 782.1 优于域内 845.0**。
- 90%+ 是**分类头**（结构/行为），与资源头（覆盖/pinball）无关。
- 唯一真实残余：p50 覆盖 R7 0.427 vs 域内 0.556。
- 连带：Phase 18 的"乐观"基准要从 1.0 改为 0.632；**C2 需重新论证**；禁止写"低估约 60%"。
# State

## Current Stage (2026-09-18)

- 涓荤嚎涓嶅彉锛氳棰?Agent 鏈潵棰勬祴 鈫?GPU 璋冨害銆?*涓?workload 宸插垏鍒?v03**锛坄r7_workload_v03_no_run_container`锛岀Щ闄?640 涓噸澶嶈鏁板鍣ㄨ妭鐐癸級銆?
- **鏈洖鍚堬紙Windows 鎵ц锛?*锛歅hase 21 pilot 缁撴 = 棰勬敞鍐岃礋缁撴灉锛?*宸垎鎴愮珛**锛夈€?
  - S_* `task_context` 淇锛?6 瑙嗛 paired masked/fixed锛?*FAIL 鍏ㄩ儴鍥涙潯 calibration gate**锛?
    fixed p50 R=0.397锛堥棬闄?[0.70,1.30]锛夈€乧ov p50/p90/p95 = 0.396/0.844/0.916銆乸50 pinball **鏄捐憲鎭跺寲 +51.6ms [27.2,74.4]**銆?
  - 绠＄嚎 gate 鍏ㄨ繃锛坖oin 1.0銆乽nknown_rate 0銆乣prefix_hash` 975/975 涓€鑷淬€乧heckpoint SHA 鏈彉锛夆啋 **骞插噣鍗曞彉閲忓鐓?*銆?
  - 鈫?**涓嶉噸鐢熸垚 S_* 棰勬祴鍖呫€佷笉閲嶈窇璋冨害鏃?*锛堣缁撹瀵?鍐呭瀛楁"浠嶆垚绔嬶級銆?
- **鍚屾棩浜嬪悗浠ｇ爜瀹℃煡锛堟渶缁堢粨璁猴級锛歝ontext 鍧楁棤缂洪櫡锛孭hase 21 鎬绘嫭缁撹鎴愮珛**
  - 鏈熼棿鏇炬彁鍑?涓変釜 stack 瀛楁琚?null 閬斀 鈫?100% UNK"鐨?P0锛?*缁忓鏌ヤ负璇姤骞跺凡鎾ゅ洖**銆?
  - 鏍瑰洜锛氭帰閽堢敤 `dict.get(field)`锛?*鏃犳硶鍖哄垎"閿笉瀛樺湪"涓?閿瓨鍦ㄤ絾鍊间负 null"**锛?閿瓨鍦ㄤ笖涓?null"鏄帹鏂€岄潪瀹炴祴銆?
  - 鍐冲畾鎬у弽璇侊紙鏄惧紡娴嬮敭瀛樺湪鎬э級锛歚task_context` 鐨勯敭闆?975/975 =
    {answer_type, domain, official_task_type, question_type, required_modalities, sub_category, temporal_scope}
    鈥斺€?鎭板ソ 7 涓敭锛屼笌 `_task_context()` 杈撳嚭鍙婂煙鍐?J 甯冨眬**瀹屽叏涓€鑷?*锛?
    `baseline`/`model_stack_id`/`planner_model_id` 鐨?`key_present = False`锛?75/975锛?
    鈫?缂栫爜鍣ㄦ甯歌蛋 `else` 鍒嗘敮銆佷粠 `stack_context` 璇荤湡瀹炲€硷紙`baseline = langgraph_react`锛?*in_vocab**锛夈€?
  - 鐪熸娈嬩綑锛堝潎闈?bug锛夛細`temporal_scope = "unknown"` 鈫?UNK锛?*registry 鏃犳瀛楁**锛屾暟鎹竟鐣岋級锛?
    `model_stack_id = stack_a_qwen3_vl8b_yolo11x` 鈫?OOV锛?*鏁版嵁闆嗗懡鍚嶆紓绉?*锛屽煙鍐呬负杩戜技鍊?`stack_a_qwen3_vl8b`锛夛紝
    鍙︿竴鍗?486 涓敋鐐癸紙`stack_b_..._yolo26n`锛夊湪璇嶈〃鍐咃紱`planner_model_id` 鍩熷唴璇嶈〃鍙湁 `unknown`锛堜粠鏉ヤ笉鍚俊鎭級锛?
    `required_modalities` 瀛楃涓插舰鎬佸煙鍐呬篃瀛樺湪锛圝 train 3439 琛屾槸瀛楃涓诧級銆?
  - **鏈仛浠讳綍绠＄嚎鏀瑰姩**锛堜笉闇€瑕侊級锛涙暀璁凡鐧昏锛氬璁″繀椤绘樉寮忔祴閿瓨鍦ㄦ€э紝涓嶅緱鐢?`.get()` 鎺ㄦ柇銆?
- **鐙珛 GPT 瀹℃煡宸插畬鎴愶紙Sol + High锛?*锛?*鏍稿績璐熺粨鏋?PASS + 璇姤鎾ゅ洖 PASS**锛涗絾椹冲洖"涓嶅瓨鍦ㄥ悓绫绘湭淇緭鍏ョ己鍙?
  涓?璇樊涓昏鏉ヨ嚜缁撴瀯閿欓厤/鍒嗗竷鍋忕Щ"锛堝凡鎾ゅ洖锛夈€傚綊妗?`docs/research/2026-09-18_fas_phase21_review_gpt.md`銆?
  - **瀹冭姹傜殑涓や釜灏佸瓨鍓嶆鏌ユ垜宸茶窇瀹屽苟鍏ㄩ儴閫氳繃**锛氱湡鍊奸『搴忕瓑浠锋€?640/640 + 64/64 妯℃澘瀹屽叏涓€鑷?
    锛圦1 杞?VERIFIED锛宲ilot 鏃犻渶閲嶈窇锛夛紱涓?pack 鍧?`min_steps=5`锛圥0-2 鍏抽棴锛夈€?
  - **鏂板叧閿偣**锛歚__UNK__` 鈮?璁粌鏃剁殑瀛楅潰閲?`"unknown"` token 鈫?`planner_model_id` / `model_stack_id`(stack_a) /
    `temporal_scope` 浠嶆槸鐪熷疄鐨?deployment input mismatch銆?
  - **闆舵垚鏈垎灞傦紙鐜版湁杈撳嚭锛?*锛歩n-vocab 鐨?stack_b 鏍″噯鏄庢樉濂戒簬 OOV 鐨?stack_a
    锛坒ixed R50 0.5820 vs 0.1751锛沜ov90 0.9012 vs 0.7878锛沜ov95 0.9699 vs 0.8620锛夆啋 鍛戒腑"鍏堟煡 stack canonicalization"銆?
    浣?*鏈夋贩娣?*锛堜袱鑰呮湰灏辨槸涓嶅悓鎵ц鏍堛€乺untime 鍒嗗竷涓嶅悓锛夛紝闇€閰嶅鍙嶄簨瀹炴墠鑳藉畾鍥犳灉銆?
- **涓嬩竴姝ワ紙GPT 鎸囧畾锛屽潎鍑犲崄绉掔骇锛?*锛?1-B1 鎶?`planner_model_id` 璁句负瀛楅潰閲?`"unknown"`锛?
  21-B2 鎶?`stack_a_qwen3_vl8b_yolo11x` 鏄犲皠鍒拌瘝琛ㄥ唴鐨?`stack_a_qwen3_vl8b`锛?*鍓嶇疆锛氬厛纭 stack identity 瀹氫箟鐩稿悓**锛夈€?
  鍒ゆ嵁锛歮ean pinball 鏀瑰杽 鈮?0% + CI upper < 0 + coverage/R50 鏈濈洰鏍囩Щ鍔ㄣ€備袱鑰呴兘澶辫触鎵嶅綊鍥犲埌鍒嗗竷鍋忕Щ銆?
- **寰呯敤鎴峰喅绛?*锛氣憼 鏄惁鎵ц 21-B1 / 21-B2锛堝墠缃殑 stack identity 纭锛夛紱鈶?鏂瑰悜 A锛圥2 sweep + v03-confirm300锛?
  B锛圥hase 17 濂戠害淇锛? C锛圚10-lite 绛夛級銆?
- 鍐犲啗涓庢満鍒惰〃杩颁粛鎸?09-17 鐘舵€侊細**v03 涓?`predopt_h5_q95`锛?r95锛変负褰撳墠鍐犲啗**锛屾満鍒?= "涓庢湭鏉ユ瀵归綈鐨勪繚瀹堝熬閮ㄨ仛鍚?銆?

## Current Stage (2026-09-17)

- 涓荤嚎涓嶅彉锛氳棰?Agent 鏈潵棰勬祴 鈫?GPU 璋冨害锛涘啝鍐涗粛鏄?r95/q95锛堥€愭楠?runtime p95 姹傚拰锛夈€?
- **09-16 鈫?09-17 鏂板**锛?
  - Phase 11鈥?5锛歳untime-only 姝ｄ氦鏃忥紙C2 涓ユ牸妫€楠岄€氳繃锛夈€丆VaR 閲嶅啓锛堝浐瀹氶暱搴﹀悗鍦烘櫙鏃忎粛钀藉悗 15鈥?7.5s锛夈€?
    浼樺寲鍨嬪弬鑰冮噸璺戯紙exploratory negative锛屼粛钀藉悗 ~19s锛夈€丆P-RHO executed 鍙橀噺淇 + 300 闆嗛厤瀵广€?
  - **trace 渚濊禆鍋囪琚惁**锛氭湭绾︽潫 MOM run 灞傚垎閲?鈭?.139锛圕I 鍏ㄨ礋锛夈€佹埅鏂?ICC=0銆佸熬閮?lift 鍙屼晶涓嶆樉钁?
    鈫?鏀惧純"鍏卞崟璋?灏鹃儴鍏卞姩"鐗╃悊鍙欎簨锛屾敼鎸?**forecast-error-aware ranking surrogate**銆?
  - 鍏紑浠撳簱涓よ疆瀹￠槄鐨?P0/P1/P2 鍏ㄩ儴钀藉湴锛堣 `POSTFIX_REPORT.md`锛夈€?
  - **Phase 16锛堟湰鍥炲悎锛?*锛歰racle 瀵圭収 confounded 璁ゅ畾 + 鎾ゅ洖锛涙柊澧?`sameshape_h5_{p50,p95,truth}` 涓夎噦锛坥pt-in锛屾湭璺戯級銆?
- **09-15 娈佃惤浠ヤ笅浠嶆槸鏈夋晥缁嗚妭蹇収**锛屼絾鍏?涓嬩竴姝?鍒楄〃宸茶繃鏃讹紙鍘嬪姏 sweep 瀹炰负宸插畬鎴愶紝6/6 cell 鏄捐憲涓烘锛夈€?

## Current Stage (2026-09-15)

- 涓荤嚎锛氳棰?Agent 宸ヤ綔娴佹湭鏉ラ娴?鈫?GPU 璋冨害锛坒orecast-aware scheduling锛夛紱Phase 0鈥? 鍏ㄩ儴瀹屾垚骞跺綊妗ｃ€?
- **褰撳墠鍐犲啗閰嶇疆**锛欻5 棰勬祴鍣紙J3:seed11锛寁3.1 濂戠害锛?+ **q95 娑堣垂**锛堥€愭楠?runtime p95 鐩稿姞锛沴oad 鍒嗛噺涓庢敹鐩婃棤鍏筹級銆?
  - dev700锛?79,125ms锛?*鈭?7,229 [鈭?9,037, 鈭?5,476]** vs E2锛沵iss 鈭?.40pp銆?
  - frozen confirm300锛堥娆′娇鐢級锛?83,248 vs 202,012锛?*鈭?8,765 [鈭?1,948, 鈭?5,932]**锛宮iss 鈭?.36pp銆?
- 璇勬祴绾緥锛歷alidation 宸插垏 **700 dev / 300 frozen confirm**锛坄data/manifests/validation_split_dev700_confirm300.json`锛宻eed 20260914锛夛紱
  璋冨弬鍙湪 dev锛涘啝鍐涘彧鍦?confirm 璺戜竴娆°€傝繍琛屽櫒鏀寔 `--episode-ids-file`銆?
- 澶栭儴璇勫锛欸PT 鏂囩尞 + 璁″垝锛坄docs/research/2026-09-14_fas_phase6_gpt_literature.md`锛夛紱绯荤粺璁烘枃瀹氫綅宸插畾绋匡紙`docs/research/2026-09-15_fas_system_story_gpt.md`锛夛細涓昏础鐚?= future-action-aware scheduling system锛涙満鍒惰础鐚?= continuation/horizon 涓诲 + runtime-tail 娑堣垂锛涘叧閿柊棰栨€ц竟鐣?= 棰勬祴**鍔ㄦ€佸睍寮€銆佸皻涓嶅瓨鍦?*鐨勬湭鏉ユ帶鍒舵祦锛堝尯鍒簬 Parrot 鐨勫凡鐭?DAG锛夛紱鏈€灏忚ˉ鍏呭疄楠?= P0 鐪熷疄 GPU replay锛堟渶澶х煭鏉匡級鈫?P1 鍘嬪姏 sweep + 棰勬祴璐ㄩ噺鏁忔劅鎬?鈫?P2 H10-lite銆傛晠浜嬬嚎銆佸熀绾挎竻鍗曚笌鎶曠姒傜巼瑙?`docs/research/2026-09-15_fas_novelty_venue_gpt.md`锛氱函浠跨湡 FGCS/JPDC/ICPP/CCGrid 鐜板疄锛岃ˉ 2-GPU replay 鍚?MLSys 12%鈫?5鈥?5%銆丄TC/EuroSys 杩涘叆鍙啿鍖洪棿锛涘绋夸笁澶ф敾鍑伙紙simulator artifact / 鍗?workload / q95 璋冨弬锛夊繀椤荤敤 replay + pressure sweep + 鏈哄埗鍒嗘瀽闃插尽銆傝ˉ鍏呭疄楠屾€昏〃涓庝笁妗ｅ疄楠岄泦瑙?`docs/research/2026-09-15_fas_experiment_table_gpt.md`锛涘伐涓氱數姊?Agent 瀹氫綅涓虹浜?workload / case study锛堝喕缁撴帴鍙ｄ笌 q95锛屽彧閲嶇敓鎴?artifacts锛涙渶濂戒笌 2-GPU replay 鍚堝苟鍋氾級銆?

## 娲昏穬绾︽潫锛堝繀椤婚伒瀹堬級

- `T_final` 灏佸瓨锛沗S_*` 鍙厑璁告帹鐞嗐€佷笉鎷熷悎锛沨oldout 涓嶈繘鍏ユ寮忓喅绛栵紙鍘嗗彶 holdout 宸茬敤浜?R0/J 鍒ゅ畾锛屼繚鎸佸瘑灏佽涔夛級銆?
- 杩愯鍣ㄦ槸 resume 寮忥細鍚屼竴 `--output-dir` 浼氳烦杩囧凡鏈?`(episode_id, policy)`锛涢噸璺戝悓涓€绛栫暐璇锋崲鏂扮洰褰曘€?
- 鍙敤 artifacts锛歚outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts`锛堝甫姣忔 model_probabilities 涓庢瘡琛?length_probabilities锛?
  涓?`outputs/sstar_predictor_artifacts_sched/prediction_artifacts`锛堟棤鍒嗗竷锛夈€侶10锛歚outputs/sstar_predictor_artifacts_h10[_sched]`銆?
- 鐜锛氳В閲婂櫒 `D:\anaconda\envs\scheduler\python.exe`锛坱orch 2.6.0+cu124锛孯TX 3060 Laptop锛夛紝鎵€鏈夎皟搴﹁繍琛?`PYTHONPATH=src`銆?
- 妗ユ帴锛欳hrome for Testing (`E:\chrome-for-testing\chrome-win64\chrome.exe`) 蹇呴』甯?
  `--proxy-server=http://127.0.0.1:7897`锛堟湰鏈虹洿杩?chatgpt.com 瓒呮椂锛夛紱浼氳瘽 URL `https://chatgpt.com/c/6a9822da-...`銆?
- Git 鍙彁浜や唬鐮?schema/閰嶇疆/娓呭崟/鏂囨。/灏忓瀷鎸囨爣锛涗笉鎻愪氦瑙嗛銆佹潈閲嶃€佸師濮?trace銆佺紦瀛樸€佸ぇ鍨嬭緭鍑恒€?

## 宸查獙璇佺粨璁猴紙鎸変富棰橈級

1. **淇℃伅浠峰€?*锛氭湭鏉ョ粨鏋勯娴嬫樉钁椾紭浜庢棤鏈潵锛圗2 vs E0 鈭?7,065ms锛?000 闆嗭級锛?
   oracle 璧勬簮 鈮?闈欐€佽〃锛?83,845 vs 184,514锛夆啋 **璧勬簮棰勬祴涓嶆槸鐡堕**锛涚湡鍊煎唴瀹瑰弽鑰屾洿宸紙+14,222ms锛夛紝
   鍘绘帀妯″瀷韬唤鏇村ソ锛堚垝12,334ms锛夆啋 鍐呭/韬唤鏃犳浠峰€硷紝**闀垮害/缁堟缁撴瀯鏄富瑕佷环鍊兼潵婧?*銆?
2. **Horizon**锛氱湡鍊?H10 vs H5 = 鈭?0,231ms [鈭?1,115, 鈭?,363]锛汬10 鈮?鏃犵晫 oracle锛汬20=H10 楗卞拰 鈫?H5鈫扝10 鏄渶澶у彲閮ㄧ讲鏉犳潌銆?
3. **娑堣垂鏂瑰紡锛?6 鍙樹綋锛岀粨璁虹ǔ瀹氾級**锛?*q95锛堥€愭楠?runtime p95 姹傚拰锛夋渶浼?*锛?
   鍦烘櫙閲囨牱+CVaR锛堢嫭绔?鈭?3.2k / 鍏卞崟璋?鈭?.8~鈭?0.5k锛夈€佽嚜閫傚簲椋庨櫓锛堚垝11.4k锛夈€佹満浼氱害鏉熴€佺敓瀛樸€佺紦瀛樸€佸唴瀹瑰叏閮ㄦ洿宸紱
   **绾潎鍊?+13.8k銆乴oad 灏鹃儴锛坙d95锛?15.1k**锛況t95 鈮?q95锛?11ms锛岀粺璁＄瓑浠凤級銆?
   鈫?鏈哄埗锛氫环鍊兼潵鑷棰勬祴閾?**runtime 灏鹃儴** 鐨?瀹屽叏鐩稿叧寮?鎯╃綒锛涘 load 缁村害鎴栧钩鍧囧寲澶勭悊浼氬墛寮卞畠銆?
4. **J 绾匡紙棰勬祴鍣級**锛歳untime 鏄捐憲浼樹簬 B1锛圝2/J3 3/3 seeds锛夛紝浣?load-duration 璐寸嚎瓒婄晫锛?5.2%/+5.3%锛夆啋 鏃?Core GO锛?
   J4 瑙ｈ€﹁礋缁撴灉锛涘綊鍥狅紙蟿=.90 宸€佅?.95 濂姐€佷腑浣嶆暟鏍″噯鏇村ソ锛夊凡缁撴锛汮 绾挎敹鏉熴€?
5. **H10 閲嶈锛堟湸绱犲欢闀跨獥鍙ｏ級**锛氳礋杩佺Щ 鈥斺€?鍓?5 姝ラ€€鍖栵紙next-family 0.803鈫?.640銆乺untime pinball +62%锛夛紝璋冨害 +14.6s锛?
   闇€ H10-lite锛圚5 backbone + 鍚庢杈呭姪澶达級銆?

## 鏈喅闂涓庨闄?

- H10 璐熻縼绉绘湭淇锛圚10-lite 鏈疄鐜帮級锛沜ache-aware 娑堣垂缂?residency/reuse 鏁版嵁锛堟殏缂擄級銆?
- 鍏ㄦ槸浠跨湡瀹為獙锛堟棤鐪熷疄绯荤粺锛夛紱璁烘枃鍙鎬?鐭澘寰?GPT 璇勪及锛堝彲鑳介渶璺?workload/鏈熼檺绋冲仴鎬т笌棰勬祴鍣楁秷璐硅仈鍚堝疄楠岋級銆?
- S_* 鍩熷閫€鍖栧凡閲忓寲锛坮ole 0.912/family 0.803/runtime pinball 1,045锛夛紱娑堣垂缁撹鍙湪褰撳墠 workload/鏈熼檺璁剧疆楠岃瘉杩囥€?
- 1,000 闆嗗凡琚娆′娇鐢?鈫?渚濊禆 dev/confirm 绾緥鎺у埗杩囨嫙鍚堬紝鍚庣画鏂版秷璐瑰櫒蹇呴』鍏堝湪 dev 涓婃姤銆?

## 涓嬩竴姝ワ紙鎸変紭鍏堢骇锛?

1. 鍏堟牳楠?GPT 寮曠敤璁烘枃鐨勭湡浼?鍑哄锛圥arrot銆丳ythia銆丳BKV銆丗ATE銆丼OLA銆乂idur 绛夛級锛屽啀鏇存柊璁″垝銆?
2. H10-lite锛圚5 backbone + 鍚庢 horizon 杈呭姪澶?闄嶆潈锛宮ulti-resolution锛夈€?
3. 澶囬€夛細7-D queue-aware rollout锛堥鏈熶綆锛夈€佽法鏈熼檺/位 绋冲仴鎬с€侀娴嬪櫒 seed 绋冲仴鎬э紙J3 seed22/33锛夈€?

## ChatGPT Web 缁戝畾锛堟潈濞佹簮锛?026-09-18 浠庡綊妗ｆ仮澶嶏級

2026-09-15 鍘嬬缉鏃惰鍧椾粠娲昏穬 STATE 涓㈠け锛屼粎瀛樺綊妗ｏ紱鐜版仮澶嶅苟鏇存柊銆俿kill 瑙勫畾姝ゅ潡鏄細璇濈粦瀹氱殑鍞竴鐪熺浉婧愩€?

```yaml
chatgpt_web:
  status: active
  generation: 7
  conversation_id: "6a9822da-e278-83e9-9c1a-675923acda0e"
  conversation_url: "https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e"
  title: "鏋舵瀯璁捐璇勪及"
  model: "GPT-5.6 Sol"
  reasoning: "High"
  created_at: "2026-09-02 21:30 CST"
  last_verified_at: "2026-09-08 17:19 CST"
  last_used_at: "2026-09-08 17:19 CST"
  parent_conversation_id: null
  last_rollover_reason: null
```

**2026-09-18 瀹炴祴锛圵indows 渚фˉ鎺ワ級**锛氫細璇濆彲杈俱€佺嚎绋嬪仴搴凤紙鏈疆涓?Phase 21 璁″垝锛夛紝妯″瀷閫夋嫨鍣ㄥ綋鍓嶆樉绀?
**`DS Flash`** 鑰岄潪鐧昏鐨?`GPT-5.6 Sol` 鈫?model gate 鎷掔粷鎻愪氦锛坄selected_model: null`銆乣high_selected: false`锛夛紝
**鏈彁浜や换浣?brief**銆傚湪鐢ㄦ埛鎶婅浼氳瘽鐨勬墜鍔ㄦā鍨嬪垏鍥?`GPT-5.6 Sol` + High 涔嬪墠锛岀綉椤靛鏌ヤ笉鍙敤銆?

## 璁板綍鎸囬拡

- 瀹為獙璁板綍锛歚experiments/EXP-20260911_forecast_aware_scheduling/PHASE{0..7A,7BC}_REPORT.md`锛?
  `experiments/EXP-20260911_p9d_j_*`锛圧0/J/J4/H10/predictor acceptance锛夈€?
- 鏂囩尞/璇勫锛歚docs/research/2026-09-14_fas_phase6_gpt_literature.md`銆乣docs/research/2026-09-11_fas_*`銆乣docs/p9d_*`銆?
- 鎺у埗闈㈠巻鍙诧紙鏈褰掓。锛夛細`.project/archive/2026-09-15_pre_compact/`銆?
- 闂ㄧ锛歚.project/EXPERIMENT_GATE.json`锛堟椿璺冨疄楠?+ 鎸囬拡锛夈€?



