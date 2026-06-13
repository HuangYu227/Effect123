# Current Framework Rationality and Short-cycle Validation Plan

## 1. 当前框架相比旧版本的核心进步

旧版本最大的问题不是“没有文本输入”，而是文本进入生成器的方式太容易被主干忽略。V6.1 的主路径是 `caption -> text_encoder -> mean-pooled text_context -> regime/global gate/operator bank`。这条路径里，文本在进入生成器前已经被压成一个全局向量，后续主要通过 concat、MLP、全局 gate 或 broadcast FiLM 影响速度场。它能提供样本级条件，但很难回答“文本中的哪类语义正在控制哪段时间、哪个通道、哪个频段”。

V6.2 的关键进步是把文本条件从“静态全局向量”改成“生成状态内的跨模态交互”。当前路径变成：`caption -> slot_tokens`，同时从当前 flow state `x_t` 构造 time patch tokens、channel summary tokens、spectral band tokens；然后 text tokens 与 state tokens 做双向 cross-attention，得到 `bridge_context [B,D]` 和 `expert_context [B,3,D]`。这意味着文本不是在生成前被一次性压缩，而是在每次 velocity prediction 时和当前噪声序列状态发生交互。

这个变化直接针对旧版本的四个薄弱点：

- 文本影响弱：旧版 `text_context` 对所有时间、通道、expert 近似共享；新版 `expert_context` 分别进入 temporal/channel/frequency expert，使文本至少在机制层面产生差异化调制。
- 模块名义大于实际：旧版 `LatentRegimeAdapter` 如果 posterior 接近均匀，就很难证明 regime 真的学到动态结构；新版 bridge 的中间量可被直接检查，包括 text-to-state attention entropy、state-to-text entropy、bridge context norm、expert context norm。
- loss 过散：旧路线容易引入 trend/frequency/volatility 等伪标签式辅助项；新版仍以 CFM 为主，只保留一个可选 batch-level text-state InfoNCE，对齐的是真实配对文本和当前生成状态，不引入手工细粒度标签。
- 生成器可能忽略文本：旧版 global gate 和 operator bank 都能只依赖 `x_t` 的统计特征工作；新版 gate 使用 bridge 后的文本状态混合条件，operator experts 也接收 bridge 输出。如果 normal/shuffled/blank 的差距仍然很小，就能明确定位为 bridge/text encoder 还不够，而不是“结构图上看起来有文本但不可诊断”。

因此，V6.2 不是简单多加一个模块，而是把 text-to-series 的控制点从“后期全局注入”前移到“velocity field 内部的 text-state 交互”。这比旧版本更符合条件生成任务的因果链：文本语义先和当前序列形态对齐，再决定 operator gate 和各 expert 的速度修正。

## 2. 当前框架相比已有方法的关键差异与可写成论文创新点的地方

本地参考实现显示，优秀方法背后的共同点不是“模块多”，而是条件信息被放在生成主干能真正改变输出的位置。

VerbalTS 的 `TextProjectorMVarMScaleMStep` 会把文本投影到变量、尺度、diffusion step 相关的条件张量，再在 residual diffusion block 中用 add、cross-attention 或 AdaLN 注入。这比单纯 concat 强，因为条件直接进入扩散残差块。我们吸收了这一点：条件不能只在输入端出现。但我们没有照搬它的 var/scale/step projector，因为当前任务不是只有固定变量/尺度模板，且我们不想让文本直接预测过细 patch 元信息。V6.2 选择从 `x_t` 动态构造 time/channel/spectral state tokens，让文本和当前生成状态对齐，再输出 expert-specific context。

TimeCMA 的 CrossModal 层说明显式 q/k/v 跨模态注意力比 add/concat 更合理。但 TimeCMA 的目标偏向表征/预测，cross-modal alignment 不是直接服务于生成速度场。V6.2 的差异是把 cross-attention 放在 flow matching 的 denoiser/velocity predictor 里，每个采样时间 `t` 都重新基于 `x_t` 形成条件，这更贴合生成任务。

T2S/DiT 风格的 rectified flow 和 AdaLN 说明主生成目标可以保持简单，条件通过 block modulation 进入主干即可。T2S 代码里 `c = t + text_input` 再驱动 AdaLN，这条路径稳定但仍偏全局；我们的 bridge 保留了 rectified/CFM 的简洁目标，同时把条件拆成全局 gate context 和 temporal/channel/frequency expert context，避免所有生成机制共享同一个文本向量。

CNDiff/DiT 类条件扩散把数值条件或历史序列通过 condition network 注入 denoiser，优点是条件路径短、梯度清楚；但它不是文本条件，也没有 text-series 语义对齐问题。V6.2 的价值是把“短条件路径”保留下来，同时显式处理文本 token 与序列状态 token 的匹配。

Diff-MoE 的启发是 expert 不应只是并列卷积层，而应和生成阶段或空间/token 特征相关。V6.2 不是通用 MoE，而是把 experts 固定为时间段、通道交互、频率带三类时间序列机制，并用 bridge 产生 expert-specific condition。这可以形成比“generic MoE for time series”更清晰的论文叙事：文本语义不是选择抽象专家，而是调制可解释的时序生成机制。

可写成论文创新点的叙事应当收敛为三句话：

1. State-aware textual velocity conditioning：文本条件在每个 flow state 内与 time/channel/spectral tokens 交互，而不是作为静态 prompt embedding 后期拼接。
2. Mechanism-aware expert modulation：cross-modal bridge 输出分别调制 temporal、channel、frequency velocity experts，使文本语义控制不同时间序列形态机制。
3. Minimal paired alignment objective：主目标仍是 CFM，只用可选 paired text-state InfoNCE 校准 bridge，不依赖手工形态伪标签。

这套设计有价值，但它还不是无条件强于所有方法。当前最大的短板是 LongCLIP 路径返回的是 caption-level/slot-level embedding；如果 `max_caption_slots=1`，text side 只有一个 slot，bridge 仍然不是 word-token 级对齐。它比旧版强，因为 state token 侧已经细粒度化，并且 context 会进入 expert；但如果未来要冲顶会，需要考虑让 text encoder 暴露 token hidden states 或使用 caption candidates 提高文本 token 数。

## 3. 当前框架仍然可能存在的风险，尤其是短时间训练不出效果的原因

第一，短训练阶段 bridge 可能还没有学会稳定使用文本。CFM 主损失会优先学习数据分布和速度幅度，早期 FID/JFTSD 可能先改善，而 CTTP 或 normal-shuffled 差距滞后出现。因此 2800 step 指标差不能直接否定结构，但如果 bridge attention、expert context norm、operator usage 全部退化，就说明文本路径没有激活。

第二，当前 bridge 是 state-aware，但 `x_t` 在大部分 flow time 上含有较强噪声。早期训练中 state tokens 可能更多反映噪声统计，而不是目标形态。这个风险可以通过分 flow time 的诊断来确认：如果 bridge attention 在所有 `t` 上都近似均匀，说明交互没有学到；如果在中后期 `t` 更集中，结构仍有潜力。

第三，InfoNCE alignment loss 可能有副作用。天气 caption 之间语义相似，batch 内负样本不一定是真负样本；权重过大会让 bridge 学“区分样本”而不是学“控制形态”。所以 `v62_bridge_align` 只是 ablation，不应默认作为主创新。若它提升 CTTP 但恶化 FID/JFTSD，要优先保留 bridge-only。

第四，global operator gate 仍然是样本级 gate。局部事件的精细控制主要依赖 expert 内部的 text-conditioned features，而不是 gate 本身。如果可视化显示全局趋势有改善但局部 spike/phase/control 仍弱，下一步最小修改不是加新 loss，而是把 bridge 输出扩展为 time-local context map。

第五，当前 text encoder 冻结且领域迁移有限。LongCLIP 对自然语言长文本有效，但不一定天然理解 weather caption 中的周期、波动、变量关系。如果 normal/shuffled 差距一直小，同时 bridge diagnostics 正常，问题可能在文本编码粒度，而不是生成主干。

第六，短周期指标本身有噪声。FID/JFTSD 更看生成分布，CTTP 更看文本对齐；三者可能不同步。判断“框架更好”不能只看单点指标，而要看 caption sensitivity、bridge diagnostics、velocity health 和曲线形态是否一致改善。

结构层面的评价标准如下：

- 文本深度参与：`bridge_context_norm`、`bridge_expert_context_norm` 非零且稳定，normal 与 blank 的 expert/gate 行为不同。
- 条件路径短且梯度有效：`bridge_alignment_loss` 可下降，`vCos` 提升，`pred_target_rms_ratio` 不长期接近 0。
- 存在可学习交互：`bridge_text_to_state_entropy_norm` 不长期等于 1，`bridge_*_max_prob` 不长期等于均匀值。
- 生成器难以忽略文本：normal 明显优于 shuffled/blank，尤其是 CTTP 上升、JFTSD 下降。
- 模型复杂度适合短训：bridge-only 应先超过 `v61_noregime`；如果 align 版本更慢但无收益，关闭 alignment。

短周期看点：

- 2800 step：只做机制健康检查，不急着判最终指标。看 `vR` 是否从极低值恢复、`vCos` 是否为正、bridge entropy 是否非 NaN、expert context 是否进入 operator_aux、normal/shuffled/blank 是否开始有微弱差距。
- 5000 step：看 caption sensitivity 是否成形。若 FID/JFTSD 还一般，但 normal CTTP 已高于 shuffled/blank，说明文本路径有效；若三者完全一致，优先怀疑文本条件仍被忽略。
- 10000 step：看综合质量。`v62_bridge` 应至少接近或超过 `v61_noregime`，并在 normal-shuffled/blank 差值上更强；`v62_bridge_align` 只有在不损害 FID/JFTSD 的情况下才保留。

建议额外做四类验证：

- 文本打乱：同一 checkpoint 下 normal vs shuffled，观察 CTTP/JFTSD 差值。
- 条件置零：normal vs blank，确认模型不是只靠无条件分布。
- 文本相似性分组：把 caption 按 trend/periodic/spike/volatility 关键词粗分组，只做评估分桶，不作为训练标签；看生成曲线统计是否随组变化。
- 曲线可视化：每组抽 16 条 normal/shuffled/blank 生成曲线，同一 noise seed 对比形态差异，重点看趋势方向、周期强度、局部事件位置、通道相关性。

## 4. 有限资源下的 014 三张卡实验设计指令

三张卡的分工应围绕“bridge 是否真的带来增益”而不是同时探索太多变量。

GPU 0 跑主实验 `v62_bridge`：只启用 cross-modal bridge，不启用 alignment loss，不启用 latent regime。它回答“结构交互本身是否比旧版强”。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_weather.py \
  --config configs/weather_v62_bridge.yaml \
  --data-root /home/newuser001/huangyu/Research/Effect123/datasets/synth-m \
  --text-encoder-mode longclip \
  --text-encoder-model /home/newuser001/huangyu/Research/Effect123/save/Longclip \
  --batch-size 1024 \
  --eval-batch-size 512 \
  --max-steps 10000 \
  --checkpoint-dir checkpoints/v62_bridge_gpu0 \
  > logs/v62_bridge_gpu0.log 2>&1 &
```

GPU 1 跑 `v62_bridge_align`：同样 bridge，但加 `bridge_alignment_weight=0.001`。它回答“唯一辅助 alignment loss 是否有必要”。

```bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_weather.py \
  --config configs/weather_v62_bridge_align.yaml \
  --data-root /home/newuser001/huangyu/Research/Effect123/datasets/synth-m \
  --text-encoder-mode longclip \
  --text-encoder-model /home/newuser001/huangyu/Research/Effect123/save/Longclip \
  --batch-size 1024 \
  --eval-batch-size 512 \
  --max-steps 10000 \
  --checkpoint-dir checkpoints/v62_bridge_align_gpu1 \
  > logs/v62_bridge_align_gpu1.log 2>&1 &
```

GPU 4 跑严格旧版对照 `v61_noregime_control`：关闭 latent regime、无 bridge、保留 structural operator + global gate。它回答“V6.2 的提升是否来自 bridge，而不是 operator/gate 本身”。

```bash
CUDA_VISIBLE_DEVICES=4 nohup python scripts/train_weather.py \
  --config configs/weather_v61_regime_noregime.yaml \
  --data-root /home/newuser001/huangyu/Research/Effect123/datasets/synth-m \
  --text-encoder-mode longclip \
  --text-encoder-model /home/newuser001/huangyu/Research/Effect123/save/Longclip \
  --batch-size 1024 \
  --eval-batch-size 512 \
  --max-steps 10000 \
  --checkpoint-dir checkpoints/v61_noregime_control_gpu4 \
  > logs/v61_noregime_control_gpu4.log 2>&1 &
```

短周期 checkpoint 处理：

- 2800 step：训练脚本每 200 step 更新 `latest.pt`，但固定 step 文件每 1000 step 保存一次；日志到 2800 时立刻复制一份 `latest.pt` 作为 `step_00002800.pt`。
- 5000 step：直接用 `step_00005000.pt`。
- 10000 step：用 `step_00010000.pt` 或 `latest.pt`。

```bash
cp checkpoints/v62_bridge_gpu0/latest.pt checkpoints/v62_bridge_gpu0/step_00002800.pt
cp checkpoints/v62_bridge_align_gpu1/latest.pt checkpoints/v62_bridge_align_gpu1/step_00002800.pt
cp checkpoints/v61_noregime_control_gpu4/latest.pt checkpoints/v61_noregime_control_gpu4/step_00002800.pt
```

评估时每个 checkpoint 跑 normal、shuffled、blank。下面给出模板，三个实验只替换 `CUDA_VISIBLE_DEVICES`、config、checkpoint 和输出名：

```bash
# normal
CUDA_VISIBLE_DEVICES=0 python scripts/eval_verbalts_metrics.py \
  --config configs/weather_v62_bridge.yaml \
  --checkpoint checkpoints/v62_bridge_gpu0/step_00005000.pt \
  --data-root /home/newuser001/huangyu/Research/Effect123/datasets/synth-m \
  --verbalts-root /home/newuser001/huangyu/Research/VerbalTS \
  --clip-folder /home/newuser001/huangyu/Research/VerbalTS/save/synth-m_cttp \
  --text-encoder-mode longclip \
  --text-encoder-model /home/newuser001/huangyu/Research/Effect123/save/Longclip \
  --split test --n-samples 10 --batch-size 512 \
  > logs/eval_v62_bridge_5000_normal.json

# shuffled
CUDA_VISIBLE_DEVICES=0 python scripts/eval_verbalts_metrics.py \
  --config configs/weather_v62_bridge.yaml \
  --checkpoint checkpoints/v62_bridge_gpu0/step_00005000.pt \
  --data-root /home/newuser001/huangyu/Research/Effect123/datasets/synth-m \
  --verbalts-root /home/newuser001/huangyu/Research/VerbalTS \
  --clip-folder /home/newuser001/huangyu/Research/VerbalTS/save/synth-m_cttp \
  --text-encoder-mode longclip \
  --text-encoder-model /home/newuser001/huangyu/Research/Effect123/save/Longclip \
  --split test --n-samples 10 --batch-size 512 --caption-shuffle \
  > logs/eval_v62_bridge_5000_shuffled.json

# blank
CUDA_VISIBLE_DEVICES=0 python scripts/eval_verbalts_metrics.py \
  --config configs/weather_v62_bridge.yaml \
  --checkpoint checkpoints/v62_bridge_gpu0/step_00005000.pt \
  --data-root /home/newuser001/huangyu/Research/Effect123/datasets/synth-m \
  --verbalts-root /home/newuser001/huangyu/Research/VerbalTS \
  --clip-folder /home/newuser001/huangyu/Research/VerbalTS/save/synth-m_cttp \
  --text-encoder-mode longclip \
  --text-encoder-model /home/newuser001/huangyu/Research/Effect123/save/Longclip \
  --split test --n-samples 10 --batch-size 512 --blank-captions \
  > logs/eval_v62_bridge_5000_blank.json
```

对 GPU 1 的 `v62_bridge_align`，把 config 改成 `configs/weather_v62_bridge_align.yaml`，checkpoint 改成 `checkpoints/v62_bridge_align_gpu1/...`，输出名前缀改成 `eval_v62_bridge_align_*`。对 GPU 4 的旧版对照，把 config 改成 `configs/weather_v61_regime_noregime.yaml`，checkpoint 改成 `checkpoints/v61_noregime_control_gpu4/...`，输出名前缀改成 `eval_v61_noregime_control_*`。

最终判定规则：

- 如果 `v62_bridge normal` 明显优于 shuffled/blank，而 `v61_noregime_control` 差距小，说明 bridge 是有效创新。
- 如果 `v62_bridge_align` 比 `v62_bridge` 只提升 CTTP 但 FID/JFTSD 变差，alignment loss 不作为主实验。
- 如果三者 normal/shuffled/blank 都接近，问题不在 regime，而在文本编码粒度或 bridge 强度；最小后续修改是增加 text token 粒度，而不是加更多 loss。
- 如果 `v62_bridge` 生成质量差但 caption sensitivity 强，说明条件路径有效、生成能力不足；后续优先调训练稳定性和 expert modulation scale。
