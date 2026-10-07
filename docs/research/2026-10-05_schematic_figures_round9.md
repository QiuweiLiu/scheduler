# 第 9 轮:三张示意图的可渲染规格与 SVG(GPT 回复全文)

> 来源:网页版 GPT(GPT-5.6 Sol + High;generation 11 会话 `6ac1f777-6728-83ec-9281-18a3cb787f23`),
> 2026-10-05 第 9 轮。原始回复全文照录(含三张图的 SVG 代码)。

炜，下面直接给三张图的**可渲染规格**，每张都包含：

1. 画布尺寸建议  
2. 元素清单（英文标签，逐字）  
3. 布局说明  
4. 配色说明  
5. 一句话 caption  
6. **可直接渲染的 SVG 代码**

你可以直接把 SVG 保存成 `.svg`，再转成 PNG；也可以照着用 matplotlib 复刻。

---

# Fig.1 问题示意图  
## 主题
**Decision happens before future stages are revealed**

---

## 1) 画布尺寸建议
- `1200 x 560`
- `viewBox="0 0 1200 560"`

---

## 2) 元素清单（英文标签）
- 标题（可选，不放也行）  
  - `"Scheduling decision is made before the future is revealed"`
- 顶部区域标签  
  - `"Known prefix"`
  - `"Unrevealed suffix"`
- 工作流节点  
  - `"Planner"`
  - `"Spatial Tool"`
  - `"?"`
  - `"?"`
  - `"Answer"`
- 分界标注  
  - `"decision time"`
- 决策框  
  - `"Residency / loading decision"`
- GPU 盒子  
  - `"GPU 0"`
  - `"GPU 1"`
  - `"Resident models"`
  - `"Planner-8B"`
  - `"Spatial-8B"`
  - `"Answer-3B"`
  - `"YOLO"`
- 辅助文字（可选）
  - `"prefix is visible now"`
  - `"future stages are still unknown"`

---

## 3) 布局
- 上半部分：一条横向工作流链
- 中间竖向虚线：已知前缀 / 未知后缀分界
- 分界处往下一个琥珀色箭头，指向中部的决策框
- 下半部分：左右两个 GPU 卡槽，显示 resident models
- 重点传达：**调度器现在必须决策，但未来两步还只是问号**

---

## 4) 配色
- 已知节点：深蓝 `#2b5d8a`
- 当前可见工具节点：青绿 `#2f8f83`
- 未知节点：灰 `#6b6b6b`
- 决策箭头 / 决策框强调：琥珀 `#c98a2e`
- GPU 框及浅底：`#dfe6ee`
- 说明文字：灰 `#6b6b6b`

---

## 5) Caption
**Fig. 1. The scheduler must make residency and loading decisions at the current decision time, even though the downstream workflow suffix has not yet been revealed.**

---

## 6) SVG 代码
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 z" fill="#6b6b6b"/>
    </marker>
    <marker id="arrowAmber" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 z" fill="#c98a2e"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1200" height="560" fill="white"/>

  <!-- Top labels -->
  <text x="230" y="55" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#2b5d8a" text-anchor="middle">Known prefix</text>
  <text x="830" y="55" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#6b6b6b" text-anchor="middle">Unrevealed suffix</text>

  <!-- Workflow line -->
  <line x1="120" y1="140" x2="1010" y2="140" stroke="#6b6b6b" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Nodes -->
  <rect x="90" y="110" rx="12" ry="12" width="120" height="60" fill="#2b5d8a"/>
  <text x="150" y="147" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="white" text-anchor="middle">Planner</text>

  <rect x="280" y="110" rx="12" ry="12" width="150" height="60" fill="#2f8f83"/>
  <text x="355" y="147" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="white" text-anchor="middle">Spatial Tool</text>

  <rect x="515" y="110" rx="12" ry="12" width="100" height="60" fill="white" stroke="#6b6b6b" stroke-width="2" stroke-dasharray="6 5"/>
  <text x="565" y="147" font-family="Arial, Helvetica, sans-serif" font-size="28" fill="#6b6b6b" text-anchor="middle">?</text>

  <rect x="675" y="110" rx="12" ry="12" width="100" height="60" fill="white" stroke="#6b6b6b" stroke-width="2" stroke-dasharray="6 5"/>
  <text x="725" y="147" font-family="Arial, Helvetica, sans-serif" font-size="28" fill="#6b6b6b" text-anchor="middle">?</text>

  <rect x="870" y="110" rx="12" ry="12" width="120" height="60" fill="#2b5d8a"/>
  <text x="930" y="147" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="white" text-anchor="middle">Answer</text>

  <!-- Separator -->
  <line x1="470" y1="80" x2="470" y2="240" stroke="#c98a2e" stroke-width="2" stroke-dasharray="6 6"/>
  <text x="470" y="92" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#c98a2e" text-anchor="middle">decision time</text>

  <!-- Explanatory text -->
  <text x="245" y="195" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b" text-anchor="middle">prefix is visible now</text>
  <text x="760" y="195" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b" text-anchor="middle">future stages are still unknown</text>

  <!-- Decision arrow -->
  <line x1="470" y1="240" x2="470" y2="285" stroke="#c98a2e" stroke-width="3" marker-end="url(#arrowAmber)"/>

  <!-- Decision box -->
  <rect x="360" y="290" rx="12" ry="12" width="220" height="58" fill="#fff7e8" stroke="#c98a2e" stroke-width="2"/>
  <text x="470" y="326" font-family="Arial, Helvetica, sans-serif" font-size="19" font-weight="700" fill="#c98a2e" text-anchor="middle">Residency / loading decision</text>

  <!-- GPU boxes -->
  <rect x="120" y="390" rx="16" ry="16" width="390" height="130" fill="#f7f9fb" stroke="#2b5d8a" stroke-width="2"/>
  <text x="150" y="420" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="#2b5d8a">GPU 0</text>
  <text x="150" y="445" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b">Resident models</text>
  <rect x="145" y="460" rx="10" ry="10" width="120" height="34" fill="#2b5d8a"/>
  <text x="205" y="482" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="white" text-anchor="middle">Planner-8B</text>
  <rect x="280" y="460" rx="10" ry="10" width="130" height="34" fill="#2f8f83"/>
  <text x="345" y="482" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="white" text-anchor="middle">Spatial-8B</text>

  <rect x="690" y="390" rx="16" ry="16" width="390" height="130" fill="#f7f9fb" stroke="#2b5d8a" stroke-width="2"/>
  <text x="720" y="420" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="#2b5d8a">GPU 1</text>
  <text x="720" y="445" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b">Resident models</text>
  <rect x="715" y="460" rx="10" ry="10" width="120" height="34" fill="#2b5d8a"/>
  <text x="775" y="482" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="white" text-anchor="middle">Answer-3B</text>
  <rect x="850" y="460" rx="10" ry="10" width="90" height="34" fill="#b3544a"/>
  <text x="895" y="482" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="white" text-anchor="middle">YOLO</text>
</svg>
```

---

---

# Fig.2 系统框图  
## 主题
**The policy reads predictions and observable state, not machine-response tables**

---

## 1) 画布尺寸建议
- `1320 x 620`
- `viewBox="0 0 1320 620"`

---

## 2) 元素清单（英文标签）
- 左侧输入  
  - `"Observed prefix"`
- 中间预测器  
  - `"Frozen predictor"`
  - `"(per-instance, ≤5 steps)"`
- 预测输出
  - `"Runtime T"`
  - `"Model probs P(M)"`
  - `"Length probs P(L)"`
- 两个策略模块  
  - `"Ordering"`
  - `"current + finite future cost"`
  - `"Residency control"`
  - `"keep / evict / prefetch"`
- 执行端  
  - `"GPU execution"`
  - `"GPU 0"`
  - `"GPU 1"`
- evaluator 层  
  - `"Evaluator / environment only"`
  - `"Measured substrate tables"`
  - `"slowdown"`
  - `"interference"`
  - `"batching"`
  - `"load / evict costs"`
  - `"environment only — policy never reads"`

---

## 3) 布局
- 主流程从左到右：
  `Observed prefix` → `Frozen predictor` → 两个并列模块（Ordering / Residency control）→ `GPU execution`
- predictor 下方放三个输出小框
- 最下方放一个横向虚线大框，作为 evaluator / environment 层
- evaluator 只通过虚线箭头连到 `GPU execution`
- 不从 evaluator 连到 predictor / ordering / residency
- 红色文字强调：**policy never reads**

---

## 4) 配色
- 主流程框：深蓝 / 青绿 / 浅底
- Residency 模块可用琥珀强调
- evaluator 层：灰色虚线框
- “policy never reads” 用红色 `#b3544a`

---

## 5) Caption
**Fig. 2. The scheduling policy consumes only frozen per-instance predictions and observable state; measured substrate tables are confined to the evaluator/environment layer.**

---

## 6) SVG 代码
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="620" viewBox="0 0 1320 620">
  <defs>
    <marker id="arrow2" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 z" fill="#6b6b6b"/>
    </marker>
    <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 z" fill="#2b5d8a"/>
    </marker>
    <marker id="arrowGray" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 z" fill="#6b6b6b"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1320" height="620" fill="white"/>

  <!-- Input -->
  <rect x="60" y="120" rx="14" ry="14" width="180" height="70" fill="#dfe6ee" stroke="#2b5d8a" stroke-width="2"/>
  <text x="150" y="150" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#2b5d8a" text-anchor="middle">Observed prefix</text>
  <text x="150" y="173" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b" text-anchor="middle">running instance only</text>

  <!-- Predictor -->
  <rect x="330" y="105" rx="16" ry="16" width="250" height="100" fill="#f7f9fb" stroke="#2b5d8a" stroke-width="2"/>
  <text x="455" y="145" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#2b5d8a" text-anchor="middle">Frozen predictor</text>
  <text x="455" y="172" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#6b6b6b" text-anchor="middle">(per-instance, ≤5 steps)</text>

  <!-- Outputs -->
  <rect x="350" y="245" rx="10" ry="10" width="110" height="42" fill="#2b5d8a"/>
  <text x="405" y="272" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="white" text-anchor="middle">Runtime T</text>

  <rect x="470" y="245" rx="10" ry="10" width="140" height="42" fill="#2f8f83"/>
  <text x="540" y="272" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="white" text-anchor="middle">Model probs P(M)</text>

  <rect x="620" y="245" rx="10" ry="10" width="140" height="42" fill="#c98a2e"/>
  <text x="690" y="272" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="white" text-anchor="middle">Length probs P(L)</text>

  <!-- Arrows input to predictor -->
  <line x1="240" y1="155" x2="330" y2="155" stroke="#2b5d8a" stroke-width="2.5" marker-end="url(#arrowBlue)"/>

  <!-- Ordering -->
  <rect x="760" y="100" rx="16" ry="16" width="220" height="95" fill="#f7f9fb" stroke="#2b5d8a" stroke-width="2"/>
  <text x="870" y="136" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#2b5d8a" text-anchor="middle">Ordering</text>
  <text x="870" y="166" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b" text-anchor="middle">current + finite future cost</text>

  <!-- Residency -->
  <rect x="760" y="235" rx="16" ry="16" width="220" height="95" fill="#fff7e8" stroke="#c98a2e" stroke-width="2"/>
  <text x="870" y="271" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#c98a2e" text-anchor="middle">Residency control</text>
  <text x="870" y="301" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b" text-anchor="middle">keep / evict / prefetch</text>

  <!-- Arrows predictor outputs to modules -->
  <line x1="580" y1="155" x2="760" y2="145" stroke="#6b6b6b" stroke-width="2" marker-end="url(#arrow2)"/>
  <line x1="580" y1="250" x2="760" y2="280" stroke="#6b6b6b" stroke-width="2" marker-end="url(#arrow2)"/>

  <!-- GPU execution -->
  <rect x="1050" y="130" rx="16" ry="16" width="220" height="180" fill="#f7f9fb" stroke="#2b5d8a" stroke-width="2"/>
  <text x="1160" y="165" font-family="Arial, Helvetica, sans-serif" font-size="23" font-weight="700" fill="#2b5d8a" text-anchor="middle">GPU execution</text>

  <rect x="1080" y="195" rx="10" ry="10" width="160" height="36" fill="#dfe6ee" stroke="#2b5d8a" stroke-width="1.5"/>
  <text x="1160" y="218" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#2b5d8a" text-anchor="middle">GPU 0</text>

  <rect x="1080" y="245" rx="10" ry="10" width="160" height="36" fill="#dfe6ee" stroke="#2b5d8a" stroke-width="1.5"/>
  <text x="1160" y="268" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#2b5d8a" text-anchor="middle">GPU 1</text>

  <!-- Arrows to GPU execution -->
  <line x1="980" y1="148" x2="1050" y2="185" stroke="#2b5d8a" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
  <line x1="980" y1="282" x2="1050" y2="255" stroke="#c98a2e" stroke-width="2.5" marker-end="url(#arrow2)"/>

  <!-- Environment / evaluator layer -->
  <rect x="70" y="390" rx="18" ry="18" width="1180" height="180" fill="none" stroke="#6b6b6b" stroke-width="2" stroke-dasharray="8 6"/>
  <text x="150" y="425" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#6b6b6b">Evaluator / environment only</text>
  <text x="150" y="455" font-family="Arial, Helvetica, sans-serif" font-size="18" fill="#6b6b6b">Measured substrate tables</text>

  <rect x="150" y="485" rx="10" ry="10" width="110" height="36" fill="#dfe6ee"/>
  <text x="205" y="508" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b" text-anchor="middle">slowdown</text>

  <rect x="275" y="485" rx="10" ry="10" width="125" height="36" fill="#dfe6ee"/>
  <text x="338" y="508" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b" text-anchor="middle">interference</text>

  <rect x="415" y="485" rx="10" ry="10" width="105" height="36" fill="#dfe6ee"/>
  <text x="468" y="508" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b" text-anchor="middle">batching</text>

  <rect x="535" y="485" rx="10" ry="10" width="140" height="36" fill="#dfe6ee"/>
  <text x="605" y="508" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#6b6b6b" text-anchor="middle">load / evict costs</text>

  <text x="920" y="445" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="#b3544a">environment only — policy never reads</text>

  <!-- Dashed arrow from environment to execution only -->
  <line x1="1090" y1="390" x2="1120" y2="312" stroke="#6b6b6b" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#arrowGray)"/>
  <line x1="1180" y1="390" x2="1200" y2="312" stroke="#6b6b6b" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#arrowGray)"/>
</svg>
```

---

---

# Fig.8 相关工作定位图  
## 主题
**Positioning relative to nearby systems**

---

## 1) 画布尺寸建议
- `980 x 760`
- `viewBox="0 0 980 760"`

---

## 2) 元素清单（英文标签）
- 横轴标题  
  - `"Future information: revealed  →  unrevealed"`
- 纵轴标题  
  - `"Control target: queue / batch  →  GPU residency / placement"`
- 点标签  
  - `"LLMSched"`
  - `"JITServe"`
  - `"Latency-Aware"`
  - `"Maestro"`
  - `"AgentIR"`
  - `"Ours"`
- 每个点的短注释
  - LLMSched: `"BN, no residency"`
  - JITServe: `"goodput / batching"`
  - Latency-Aware: `"device-model-based"`
  - Maestro: `"current-stage demand"`
  - AgentIR: `"IR + online profiling"`
  - Ours: `"per-instance suffix"`
- 可选角标区说明
  - `"revealed frontier"`
  - `"unrevealed suffix"`
  - `"ordering-oriented"`
  - `"residency / placement"`

---

## 3) 布局
- 二维坐标系，留白大
- 点位建议（0–1 归一化后）：
  - LLMSched `(0.43, 0.40)`
  - JITServe `(0.55, 0.30)`
  - Latency-Aware `(0.30, 0.78)`
  - Maestro `(0.23, 0.67)`
  - AgentIR `(0.67, 0.80)`
  - Ours `(0.78, 0.73)`
- Ours 用更大的圆点 + 琥珀描边
- 可轻微加横向灰色引导线，但不是必须

---

## 4) 配色
- Ours：深蓝填充 + 琥珀描边
- 其余工作：灰底蓝边或青绿/灰混合
- AgentIR 可用青绿强调（因为最接近）
- Latency-Aware 可用红色偏强调（因为 device-model-based 是一个明确差异）

---

## 5) Caption
**Fig. 8. Relative positioning: our method focuses on per-instance unrevealed suffixes and uses them for GPU residency/placement decisions, without exposing a machine-response cost model to the policy.**

---

## 6) SVG 代码
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="980" height="760" viewBox="0 0 980 760">
  <defs>
    <marker id="axisArrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 z" fill="#6b6b6b"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="980" height="760" fill="white"/>

  <!-- Axes -->
  <line x1="120" y1="640" x2="900" y2="640" stroke="#6b6b6b" stroke-width="2" marker-end="url(#axisArrow)"/>
  <line x1="120" y1="640" x2="120" y2="90" stroke="#6b6b6b" stroke-width="2" marker-end="url(#axisArrow)"/>

  <!-- Axis labels -->
  <text x="510" y="705" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#6b6b6b" text-anchor="middle">
    Future information: revealed  →  unrevealed
  </text>
  <g transform="translate(40,360) rotate(-90)">
    <text x="0" y="0" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#6b6b6b" text-anchor="middle">
      Control target: queue / batch  →  GPU residency / placement
    </text>
  </g>

  <!-- Corner hints -->
  <text x="140" y="665" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b">revealed frontier</text>
  <text x="760" y="665" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b">unrevealed suffix</text>
  <text x="138" y="620" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b" transform="rotate(-90 138 620)">ordering-oriented</text>
  <text x="138" y="180" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b" transform="rotate(-90 138 180)">residency / placement</text>

  <!-- Points -->
  <!-- Maestro -->
  <circle cx="299" cy="272" r="9" fill="#dfe6ee" stroke="#2b5d8a" stroke-width="2"/>
  <text x="315" y="268" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#2b5d8a">Maestro</text>
  <text x="315" y="289" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b">current-stage demand</text>

  <!-- Latency-Aware -->
  <circle cx="354" cy="211" r="9" fill="#fff2ef" stroke="#b3544a" stroke-width="2"/>
  <text x="370" y="207" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#b3544a">Latency-Aware</text>
  <text x="370" y="228" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b">device-model-based</text>

  <!-- LLMSched -->
  <circle cx="455" cy="420" r="9" fill="#dfe6ee" stroke="#2b5d8a" stroke-width="2"/>
  <text x="471" y="416" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#2b5d8a">LLMSched</text>
  <text x="471" y="437" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b">BN, no residency</text>

  <!-- JITServe -->
  <circle cx="549" cy="475" r="9" fill="#dfe6ee" stroke="#2b5d8a" stroke-width="2"/>
  <text x="565" y="471" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#2b5d8a">JITServe</text>
  <text x="565" y="492" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b">goodput / batching</text>

  <!-- AgentIR -->
  <circle cx="666" cy="200" r="10" fill="#e8f5f2" stroke="#2f8f83" stroke-width="2.5"/>
  <text x="682" y="196" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#2f8f83">AgentIR</text>
  <text x="682" y="217" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b">IR + online profiling</text>

  <!-- Ours -->
  <circle cx="752" cy="238" r="13" fill="#2b5d8a" stroke="#c98a2e" stroke-width="4"/>
  <text x="772" y="234" font-family="Arial, Helvetica, sans-serif" font-size="19" font-weight="700" fill="#2b5d8a">Ours</text>
  <text x="772" y="255" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#6b6b6b">per-instance suffix</text>

</svg>
```

---

# 最后给你两个实用建议

## 建议 1：统一视觉风格
你现在已有 5 张数据图，所以这 3 张示意图建议统一以下规范：
- 字体：`Arial, Helvetica, sans-serif`
- 标题字号：22–24
- 主标签字号：18–20
- 说明文字字号：14–16
- 线宽：1.8–2.2
- 圆角矩形：`rx=10~16`

## 建议 2：如果要用 matplotlib 重画
最容易重画的是：
- Fig.8（直接 scatter + annotate）
- Fig.1（rect + line + annotate）
- Fig.2（rect + arrows + dashed rectangle）

如果你愿意，我下一条可以继续直接给你：
1. **这三张图对应的 matplotlib Python 代码版本**，或者  
2. **把 SVG 再压一版成“论文排版更紧凑”的版本**。