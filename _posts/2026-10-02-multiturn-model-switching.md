---
layout: blog_detail
title: "프롬프트 라우팅 너머: 모델 선택, 대화 상태, 그리고 KV 캐시"
author: Anup Sharma (Nutanix), Stefan Wang (LinkedIn)
ext_author: Junghwan Park (박정환)
category: ["vllm-sr.ai", "translation"]
date: 2026-10-02 12:00:00
org_title: "Beyond Prompt Routing: Model Selection, Conversation State, and KV Cache"
org_link: https://vllm-sr.ai/blog/multiturn-model-switching/
---

에이전트가 모델 A에서 세 번의 턴(turn)을 마쳤습니다. 문서를 읽고, 몇 가지 사실을 저장하고, 답을 향해 작업을 시작했습니다. 네 번째 턴에는 더 강한 추론이 필요하므로, 라우터는 이 턴을 모델 B로 보냅니다.
> An agent has completed three turns on model A. It has read a document, stored
> some facts, and started working toward an answer. The fourth turn needs stronger
> reasoning, so the router sends it to model B.

요청은 성공합니다. 그런데 대화가 정말 나아졌을까요?
> The request succeeds. But did the conversation improve?

모델 B는 전체 대화 기록(transcript)을 받더라도 잘못된 답을 그대로 물려받을 수 있습니다. 모델 B의 레플리카(replica)는 점점 길어지는 프롬프트를 처음부터 다시 계산해야 할 수도 있습니다. 도구 호출이 진행 중이라면, 모델을 바꾸는 일이 그 결과를 해석하는 방식까지 바꿀 수 있습니다.
> Model B may receive the entire transcript and still inherit an incorrect answer.
> Its replica may have to compute the growing prompt again. If a tool call is in
> flight, changing models can also change how the result is interpreted.

이러한 트레이드오프를 vLLM Semantic Router, Envoy, 그리고 GPU 기반 vLLM 서버로 벤치마크했습니다. [960회 호출 홀드아웃(holdout)](https://github.com/vllm-project/semantic-router/issues/4080#issuecomment-5938865660)에서, 한 반응형(reactive) 전환 정책은 게이트를 끈 대조군(control)에 비해 강한 모델 사용 비율을 **31.8%p**, 세션당 평균 지연 시간을 **550 ms** 줄였습니다. 하지만 대조군이 통과한 최종 검사 세 건에서 실패했습니다. 별도로 진행한 네이티브 도구(native-tool) 실험에서는 어려운 경계를 만나기 *전에* 충분한 능력을 가진 모델을 고르는 것의 이점이 드러났습니다.
> We benchmarked these tradeoffs through vLLM Semantic Router, Envoy, and GPU-backed
> vLLM servers. In the [960-call holdout](https://github.com/vllm-project/semantic-router/issues/4080#issuecomment-5938865660), one reactive switching policy reduced
> strong-model use by **31.8 percentage points** and mean session latency by
> **550 ms** compared with a gate-disabled control, but failed three final checks
> that the control passed. A separate
> native-tool experiment showed the benefit of choosing a capable model *before*
> a difficult boundary.

이것이 프롬프트 분류(prompt classification) 너머에 있는 문제입니다. 작업의 나머지 부분을 해치지 않으면서 다음 턴에 맞는 모델을 고르는 일입니다. 이 문제는 기존 추론 스택에서 시맨틱 라우팅(semantic routing)이 어디에 자리 잡아야 하는지도 결정합니다.
> This is the problem beyond prompt classification: choosing the right model for
> the next turn without compromising the rest of the task. It also determines
> where semantic routing belongs in an existing inference stack.

## 대화는 프롬프트보다 많은 것을 담고 있습니다 / A conversation carries more than a prompt

대화가 모델을 바꿀 때는 네 종류의 상태가 중요합니다:
> Four kinds of state matter when a conversation changes models:

| 상태 / State | 전환 시 보존하거나 재구성해야 하는 것 / What a switch must preserve or rebuild |
|---|---|
| 기록 / History | 완전하고 올바른 순서로 정렬된 메시지와 도구 결과. 이전의 실수도 함께 옮겨집니다. / Complete, correctly ordered messages and tool results. Earlier mistakes travel too. |
| 도구 소유권 / Tool ownership | 모델이 진행 중인 호출/결과 순서를 호환되게 이어 가는 것. / A compatible continuation of the model's pending call/result sequence. |
| 제공자 관리 상태 / Provider-managed state | 저장된 컨텍스트에 대한 접근, 또는 다른 백엔드로 옮기기 전의 재구성. / Access to stored context, or reconstruction before moving to another backend. |
| KV 캐시 / KV cache | 세션 ID만이 아니라, 목적지에서 쓸 수 있는 호환 가능한 프리픽스(prefix) 계산 결과. / Compatible prefix computations available at the destination, not just a session ID. |

라우터가 소유한 Responses 기록은 전체 요청으로 확장한 뒤에는 다른 곳으로 옮길 수 있게(portable) 됩니다. 그렇다고 모든 상태를 옮길 수 있게 되는 것은 아닙니다. 진행 중인 도구 루프와 옮길 수 없는 컨텍스트는 [연속성 보호 계약(continuity protection contract)](https://vllm-sr.ai/docs/tutorials/learning/protection)에 따라 소유권 제약을 유지합니다. 또한 기록을 옮길 수 있다고 해서 그 기록이 올바르다고 보장되지도 않습니다.
> Router-owned Responses history can become portable after expansion into a full
> request. That does not make all state portable: active tool loops and nonportable
> context retain ownership constraints under the
> [continuity protection contract](https://vllm-sr.ai/docs/tutorials/learning/protection).
> Nor does portable history guarantee correct history.

## 두 가지 라우팅 결정, 두 종류의 근거 / Two routing decisions, two kinds of evidence

**Semantic Router는 논리적 모델(logical model)을 고릅니다.** 요청 신호, 모델의 능력, 정책, 그리고 설정된 연속성 보호가 어떤 모델이 작업을 맡을 수 있는지를 결정합니다.
> **Semantic Router chooses the logical model.** Request signals, capabilities,
> policy, and configured continuity protection determine which model may do the work.

**추론 라우터(inference router)는 서빙 엔드포인트(serving endpoint)를 고릅니다.** 추론 라우터는 그 모델에 허용된 풀(pool) 안에서, 상태(health), 대기열 압력(queue pressure), 캐시 가용성 같은 지원되는 신호를 사용해 스케줄링합니다.
> **An inference router chooses the serving endpoint.** It schedules within that
> model's admitted pool, using supported signals such as health, queue pressure,
> and cache availability.

이 두 선택이 충돌할 필요는 없습니다. 시맨틱 정책이 모델 B를 요구한다면, 모델 A의 웜(warm) 레플리카는 선택 대상이 아닙니다. 엔드포인트 선택기(endpoint picker)는 *B의 풀 안에서* 웜 레플리카를 우선할 수 있지만, 단지 캐시 적중(cache hit)을 위해 모델 결정을 뒤집어서는 안 됩니다.
> These choices need not conflict. If semantic policy requires model B, a warm
> replica of model A is not eligible. The endpoint picker can prefer a warm replica
> *within B's pool*, but must not override the model decision merely for a cache hit.

이 프로젝트에서 쓰는 추론 라우터 중 하나는 **llm-d** 입니다. llm-d 통합은 Gateway API 라우트와 InferencePool로 이 경계를 표현합니다. 다른 추론 라우터도 같은 계약을 따를 수 있습니다. 선택 가능한 모델 집합을 보존하고, 그 안에서 스케줄링하는 것입니다.
> One inference router used in this project is **llm-d**. Its integration uses
> Gateway API routes and InferencePools to express this boundary. Other inference
> routers can follow the same contract: preserve the eligible model set, then
> schedule within it.

## KV 캐시는 전환을 시스템 차원의 결정으로 만듭니다 / KV cache makes switching a systems decision

대화 기록은 옮겨 갈 수 있지만, 모델 A의 KV 캐시가 모델 B의 캐시가 되지는 않습니다. 그 어텐션 계산 결과는 특정 모델과 특정 토큰 시퀀스에 속합니다. 목적지에는 그 모델 자체의 호환 가능한 캐시된 프리픽스가 있어야 하고, 없다면 프롬프트를 프리필(prefill)해야 합니다.
> The transcript can travel; model A's KV cache does not become model B's cache.
> Those attention computations belong to a particular model and token sequence.
> The destination needs its own compatible cached prefix or must prefill the prompt.

같은 모델에 머물러도 재사용이 보장되지는 않습니다. 다른 레플리카에 해당 프리픽스가 없을 수 있고, 블록이 축출(evict)됐을 수 있으며, 렌더링된 프롬프트가 바뀌었을 수도 있습니다. 캐시를 공유하려면 서빙 계층의 명시적인 지원이 필요합니다. [vLLM 자동 프리픽스 캐싱(automatic prefix caching)](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)은 일치하는 프리픽스의 프리필 작업을 줄여 주지만, 새 출력 토큰을 생성하는 작업은 줄여 주지 않습니다.
> Even staying on one model does not guarantee reuse: another replica may lack the
> prefix, blocks may have been evicted, or the rendered prompt may have changed.
> Cache sharing requires explicit serving support.
> [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
> reduces matching-prefix prefill work, not the work of generating new output tokens.

선택 가능성(eligibility)과 소유권 제약을 만족하는 선택지들 사이에서, 전환은 다음 두 가지를 저울질합니다:
> Among choices that satisfy eligibility and ownership constraints, switching weighs:

```text
예상되는 작업 품질 향상 (expected task-quality gain)
  대비 (versus)
추가 프리필 + 대기열 + 인계/복구 + 모델 실행 비용
(extra prefill + queueing + handoff/repair + model execution cost)
```

긴 프리픽스를 다시 만드는 일은 어려운 단계 열 개를 앞두고는 할 만할 수 있지만, 마지막 확인 응답(acknowledgement) 하나를 위해서는 그렇지 않습니다. Semantic Router의 보호 기능은 제안된 전환을 점검하기 위해 전환 마진(switch margin), 연속성 비용, 최근 결과 게이트(recent-outcome gate)를 제공합니다. 다만 이 게이트에도 의미 있는 품질 피드백이 필요합니다. HTTP 200은 해당 작업 단계가 올바르다는 근거가 아닙니다.
> Rebuilding a long prefix may be worthwhile before ten difficult steps, but not
> for a final acknowledgement. Semantic Router's protection machinery exposes
> switch margins, continuity costs, and recent-outcome gates to check proposed
> switches. These gates still need meaningful quality feedback: HTTP 200 is not
> evidence that the task step was correct.

## 기존 게이트웨이와 모델 플릿을 위한 아키텍처 / An architecture for an existing gateway and model fleet

모델이 많고 레플리카가 수천 개인 플릿(fleet)에서는 **모델 선택** 과 **레플리카 스케줄링** 을 분리해 둡니다. 아래 요청 흐름은 모든 파드(pod)를 무차별적으로 탐색하는 대신, 하나의 요청이 모델 B를 고른 뒤 레플리카 B1을 고르는 과정을 보여 줍니다.
> For a fleet with many models and thousands of replicas, keep **model choice**
> separate from **replica scheduling**. The request flow below illustrates one
> request selecting model B, then replica B1, rather than searching every pod indiscriminately.

![에이전트가 AI 게이트웨이를 통해 기록과 도구를 보냅니다. Semantic Router는 요청과 대화 컨텍스트를 사용해 모델 B를 허용합니다. 추론 라우터는 그 풀 안에서 B1을 선택하고, 모델 A는 제외됩니다. 각 레플리카는 자체 KV 캐시를 가집니다. / An agent sends history and tools through an AI gateway. Semantic Router uses request and conversation context to admit model B. The inference router selects B1 within that pool; model A is excluded. Each replica has its own KV cache.](/assets/blog/2026-10-02-multiturn-model-switching/routing-architecture.svg){:style="width:92%"}

*그림 1: 선택 가능한 모델을 고른 뒤, 그 풀 안에서 스케줄링합니다. 이것은 논리적 경계이며, 하나의 게이트웨이 프로세스를 공유할 수도 있습니다. / Figure 1: Choose an eligible model, then schedule inside its pool. These are logical boundaries; they may share a gateway process.*

에이전트는 기록과 도구 상태를 제공합니다. 게이트웨이는 요청을 인증하고 테넌트 단위(tenant-scoped)의 대화 식별자를 보존합니다. Envoy 외부 프로세서(external processor)인 Semantic Router는 선택 가능한 모델을 고르고 연속성 보호를 적용합니다. 추론 라우터는 레플리카를 고르고, 모델 서버는 요청을 실행합니다. 응답은 에이전트로 돌아가며, 도구 실행과 다음 턴은 에이전트가 맡습니다.
> The agent supplies history and tool state. The gateway authenticates the request
> and preserves tenant-scoped conversation identity. Semantic Router, an Envoy
> external processor, selects an eligible model and applies continuity protection.
> The inference router picks a replica; the model server executes the request.
> The response returns to the agent, which owns tool execution and the next turn.

다음 세 가지 통합 방식이 이 구성을 믿을 수 있게 만듭니다:

1. **컨텍스트를 누가 소유할지 정합니다.** 턴 사이에 대화 식별자를 보존하고, 저장된 기록을 어디서 해석(resolve)할지 결정합니다. 식별 헤더가 빠진 경우도 테스트합니다. 설정된 헤더가 없으면 보호 기능이 페일 오픈(fail open, 실패 시 차단하지 않고 통과시킴)될 수 있습니다.
2. **모델과 풀 사이의 계약을 명시적으로 만듭니다.** 최종 모델 선택 결과를 게이트웨이 끝까지 전달합니다. 엔드포인트가 풀에 속하는지, 그리고 공유 풀이라면 모델/어댑터 호환성을 단언(assert)합니다. 보호 중인 소유 모델이 사용할 수 없게 되면 명시적인 복구가 필요하며, 도구 루프 도중에 임의의 폴백(fallback)으로 넘어가서는 안 됩니다.
3. **요청 전체를 추적합니다.** 라우팅 사유, 최종 모델, 엔드포인트, 토큰 사용량, 캐시 근거, 지연 시간, 작업 판정을 하나로 묶습니다. 이렇게 하면 능력 불일치를 콜드(cold) 프리픽스나 바쁜 레플리카와 구분할 수 있습니다.

> Three integration choices make this composition reliable:
> 1. **Choose who owns context.** Preserve conversation identity across turns and
>    decide where stored history is resolved. Test missing identity headers:
>    protection can fail open when its configured headers are absent.
> 2. **Make the model-to-pool contract explicit.** Carry the final model selection
>    through the gateway. Assert endpoint membership and, for shared pools,
>    model/adapter compatibility. A protected owner becoming unavailable needs
>    explicit recovery, not an arbitrary fallback halfway through a tool loop.
> 3. **Trace the whole request.** Join routing reason, final model, endpoint,
>    token usage, cache evidence, latency, and task verdict. This separates a
>    capability mismatch from a cold prefix or a busy replica.

## 실제 벤치마크가 보여 준 것 / What the live benchmarks showed

고정된(frozen) 후보 `agentic-context/multiturn-switch@2026-10-dev1` 의 홀드아웃에서는 Qwen3-0.6B와 Qwen3-8B를 사용해 **160개 세션, 960회 호출** 을 실행했습니다. 두 모델은 NVIDIA L40S 한 장을 공유하는 두 개의 vLLM 서버에서 Envoy와 Semantic Router 뒤에 배치됐습니다. 다섯 가지 정책은 각각 6턴짜리 워크로드 두 종류에 걸쳐 같은 16개의 홀드아웃 시드(seed)를 받았습니다. 두 워크로드는 순서가 있는 상태 유지(ordered state retention), 그리고 한 번의 검색(retrieval) 과제 뒤에 쉬운 확인 응답이 이어지는 워크로드입니다. 임계값은 홀드아웃을 살펴보기 전에 고정했습니다.
> The holdout for the frozen candidate, `agentic-context/multiturn-switch@2026-10-dev1`, ran
> **160 sessions and 960 calls** using Qwen3-0.6B and Qwen3-8B
> on two vLLM servers sharing one NVIDIA L40S, behind Envoy and Semantic Router.
> Each of five policies received the same 16 held-out seeds across two six-turn
> workloads: ordered state retention, and a one-off retrieval challenge followed
> by easy acknowledgements. Thresholds were frozen before inspecting the holdout.

최근 결과 게이트는 관찰된 결과가 정당화할 때까지 제안된 전환을 미룰 수 있습니다. **관찰(observe)** 모드는 판정을 기록만 하고 적용하지 않으며, **강제(enforce)** 모드는 판정을 적용합니다.
> A recent-outcome gate can delay a proposed switch until observed outcomes justify
> it. **Observe** records its verdict without applying it; **enforce** applies it.

| 정책 / Policy | 최종 턴 통과율 / Final-turn pass rate | 턴 정확도 / Turn accuracy | 강한 모델 턴 / Strong-model turns | 프리픽스 캐시 적중률 / Prefix-cache hit ratio | 세션당 평균 지연 시간 / Mean latency/session |
|---|---:|---:|---:|---:|---:|
| 처음부터 끝까지 강한 모델 / Strong model throughout | 100% | 66.7% | 100% | 80.8% | 1,793 ms |
| 처음부터 끝까지 작은 모델 / Small model throughout | 50.0% | 53.6% | 0% | 80.8% | 456 ms |
| 보호 적용, 게이트 비활성화 / Protection, gate disabled | 100% | 66.1% | 83.3% | 66.1% | 1,771 ms |
| 후보 게이트, 관찰 / Candidate gate, observe | 100% | 66.7% | 83.3% | 65.9% | 1,694 ms |
| 후보 게이트, 강제 / Candidate gate, enforce | 90.6% | 69.3% | 51.6% | 64.8% | 1,221 ms |

각 실험군(arm)에는 **32개 세션, 192회 호출** 이 들어 있습니다. **최종 턴 통과율은 작업 전체의 정확도가 아닙니다.** 상태 유지 워크로드에서는 순서가 맞는 최종 종합(synthesis)을 확인하지만, 검색 워크로드에서는 마지막 확인 응답만 확인합니다. 확인 응답은 검색이 실패한 뒤에도 통과할 수 있으며, 이 때문에 중간 턴 정확도도 중요합니다.
> Each arm contains **32 sessions and 192 calls**. **Final-turn pass rate is not whole-task accuracy:**
> it checks ordered final synthesis in state retention, but only the closing
> acknowledgement in the retrieval workload. An acknowledgement can pass after a
> failed retrieval, which is why intermediate turn accuracy matters too.

게이트를 끈 대조군과 비교하면, 강제 모드는 강한 모델 턴을 덜 쓰고 더 빨리 끝났지만 최종 검사 통과는 **32/32에서 29/32로** 떨어졌습니다. 짝지은(paired) 차이는 -9.4%p였고, 부트스트랩(bootstrap) 95% 구간은 [-21.9, 0.0]이었습니다. 턴 정확도가 나아진 것처럼 보이는 결과도 결론을 내릴 수 없었습니다. 이 후보는 품질 요구 사항을 충족하지 못해 강제 적용 대상에서 제외됐습니다.
> Compared with the gate-disabled control, enforcement used fewer strong-model
> turns and finished faster, but final checks fell from **32/32 to 29/32**. The
> paired difference was -9.4 percentage points, with a bootstrap 95% interval of
> [-21.9, 0.0]; the apparent turn-accuracy improvement was also inconclusive. The
> candidate did not meet our quality requirement and was rejected for enforcement.

이 결과는 통제된 환경에서 측정한 전환 결과이며, **시맨틱 분류기의 정확도나 프로덕션 규모의 처리량이 아닙니다.** 신호와 전환 제안은 통제했고, 호출은 순차적으로 실행했으며, 출력은 짧았습니다. 캐시 비율은 백엔드 카운터의 증가분을 세션별로 평균한 값입니다. 강한 모델 비중은 사용량을 대신 보여 주는 지표(proxy)이며, 실제로 측정한 비용 절감액이 아닙니다. 별도의 [llm-d 시뮬레이터 테스트](https://github.com/vllm-project/semantic-router/pull/4443)는 모델/풀/엔드포인트 구성을 확인하며, 실제 GPU에서의 스케줄링 성능을 확인하지는 않습니다.
> These are controlled switching measurements, **not semantic-classifier accuracy
> or production-scale throughput**. Signals and proposals were controlled, calls
> were sequential, and outputs were short. Cache ratios were backend counter
> deltas averaged per session; strong-model share is a usage proxy, not measured
> dollar savings. A separate [llm-d simulator test](https://github.com/vllm-project/semantic-router/pull/4443)
> checks model/pool/endpoint composition, not real-GPU scheduling performance.

## 대화는 언제 모델을 바꿔야 할까요? / When should the conversation change models?

실패한 최종 검사 세 건은 모두 같은 일정을 가진 상태 유지 세션에서 나왔습니다. 작은 모델이 세 턴을 맡은 뒤 강한 모델이 세 턴을 맡는 일정입니다. 작은 모델이 세 번째 턴에서 실패했고, 강한 모델은 그 답을 기록에 담은 채 작업을 이어 갔습니다. 최종 종합 결과에는 순서가 잘못된 값들이 들어 있었습니다.
> All three failed final checks came from state-retention sessions with the same
> schedule: three small-model turns, then three strong-model turns. The small
> model failed turn three; the stronger model received that answer in its history
> and continued. The final synthesis contained incorrectly ordered values.

![작은 모델이 세 번째 턴에서 실패합니다. 강한 모델은 이전의 오류를 기록에 담은 채 네 번째부터 여섯 번째 턴까지 이어 갑니다. 제안된 복구 경로는 계속 진행하기 전에 그 단계를 검증하거나 복구합니다. / The small model fails turn three. The strong model continues turns four through six with the earlier error in its history. A proposed recovery path verifies or repairs that step before continuing.](/assets/blog/2026-10-02-multiturn-model-switching/conversation-handoff.svg){:style="width:92%"}

*그림 2: 전환은 실행 주체를 바꿀 뿐, 이전의 답을 바꾸지 않습니다. 복구는 제안된 전략이며, 측정된 결과가 아닙니다. / Figure 2: A switch changes the executor, not earlier answers. Repair is a proposed strategy, not a measured result.*

이 추적 결과는 뒤늦은 상향 전환(escalation)이 손상된 상태를 그대로 끌고 간다는 해석과 들어맞습니다. 하지만 그 메커니즘을 인과적으로 분리해 낸 것은 아닙니다. 다음 단계는 검증, 상태 복구, 또는 안전한 재실행(replay)을 테스트하는 것입니다. 부수 효과(side effect)가 있는 도구에는 멱등성(idempotency) 안전장치가 필요합니다.
> The traces are consistent with late escalation carrying damaged state forward;
> they do not isolate that mechanism causally. Testing verification, state repair,
> or safe replay is the next step. Side-effecting tools need idempotency safeguards.

별도의 네이티브 도구 벤치마크에서도 시점이 중요했습니다. 짝지은 시드 8개에서, **이미 알려진 도구 경계 이전에** 강한 모델을 고르자 최종 성공률이 **37.5%에서 100%로** 올라갔습니다. 그 대가로 강한 모델 턴이 52.5%p 늘었고, 프리픽스 캐시 적중률이 11.35%p 낮아졌으며, 세션당 1,275 ms가 더 걸렸습니다. 이 테스트 픽스처(fixture)는 경계를 명시적으로 제공했으며, 실제 애플리케이션에서 그 경계를 예측하는 일은 앞으로 풀어야 할 과제입니다.
> Timing also mattered in a separate native-tool benchmark. Across eight paired
> seeds, choosing the stronger model **before a known tool boundary** improved
> final success from **37.5% to 100%**. The tradeoff was 52.5 percentage points more
> strong-model turns, 11.35 points lower prefix-cache hit ratio, and 1,275 ms more
> per session. The fixture supplied the boundary explicitly; predicting it in
> real applications remains work to do.

이 실험들을 종합하면 서로 다른 두 가지 정책이 필요해 보입니다. 어려운 작업이 시작되기 전에 적절한 소유 모델을 고르는 정책, 그리고 다음 턴에서 모델만 바꾸는 데 그치지 않고 실패한 상태를 복구하는 정책입니다.
> Together, these experiments suggest two distinct policies: select an appropriate
> owner before difficult work begins, and recover failed state rather than merely
> changing the model on the following turn.

## 자신의 스택에서 측정할 것 / What to measure in your own stack

고정된 강한 모델과 작은 모델, 턴 단위 라우팅, 연속성 보호를 적용한 라우팅을, 손대지 않은 홀드아웃이 있는 짝지은 작업에서 비교합니다. 그다음 엔드포인트 스케줄링을 현실적인 부하에서 따로 테스트합니다. 최소한 다음 항목은 확인합니다:

- **작업 정확성:** 요청 성공 여부만이 아니라, 최종 결과와 중간 오류.
- **연속성:** 도구 호출/결과 순서와 저장된 기록의 이식성(portability).
- **서빙 동작:** 선택 가능한 엔드포인트 선택, 웜/콜드 프리픽스, 첫 토큰까지의 시간(time to first token), 대기열 시간, 지연 시간 백분위수.
- **장애 복구:** 소유 모델 유실, 재시도, 도구 부수 효과의 중복 실행 방지.

> Compare fixed strong and small models, per-turn routing, and continuity-protected
> routing on paired tasks with an untouched holdout. Then test endpoint scheduling
> separately, under realistic load. At minimum, check:
> - **Task correctness:** final outcomes and intermediate errors, not just successful requests.
> - **Continuity:** tool call/result sequences and stored-history portability.
> - **Serving behavior:** eligible endpoint selection, warm/cold prefixes, time to
>   first token, queue time, and latency percentiles.
> - **Failure recovery:** owner loss, retries, and protection against duplicate tool side effects.

새 게이트는 먼저 관찰 모드로 실행합니다. 속도나 모델 사용량을 튜닝하기 전에 품질 요구 사항부터 정합니다. 이 아키텍처가 주는 이점은 책임의 분담입니다. Semantic Router는 **무엇이 작업을 맡을지** 결정하고, 추론 라우터는 **어디서 실행할지** 결정하며, 작업 수준의 평가는 이 조합이 실제로 일을 끝내는지 확인합니다.
> Run new gates in observe mode first. Set quality requirements before tuning for
> speed or model usage. The architectural payoff is a division of responsibility:
> Semantic Router decides **what should do the work**, the inference router decides
> **where it should run**, and task-level evaluation checks whether the combination
> actually finishes the job.

### 재현 가능한 재실행부터 시작하기 / Start with a reproducible replay

저장소 루트에서 Python 3.10 이상과 jq를 준비한 뒤, 요청을 보내지 않고 함께 제공되는 [코딩 에이전트 세션](https://github.com/vllm-project/semantic-router/blob/62bb0b94d2ff567ff49c2a236b7bd5270f37e897/bench/agent_session_replay.py)을 살펴봅니다:
> From the repository root, with Python 3.10+ and jq, inspect the bundled
> [coding-agent session](https://github.com/vllm-project/semantic-router/blob/62bb0b94d2ff567ff49c2a236b7bd5270f37e897/bench/agent_session_replay.py)
> without sending a request:

```bash
python3 bench/agent_session_replay.py --dry-run | \
  jq '{fixture, dry_run, tools, summary}'
```

이 명령은 도구 10개를 사용하는 요청 3개를 계획합니다. 이 요청들을 직접 운영하는 게이트웨이로 재실행하려면 다음과 같이 합니다:
> It plans three requests with ten tools. To replay them through your own gateway:

```bash
REPLAY_ID="blog-replay-$(date +%s)"
python3 bench/agent_session_replay.py \
  --base-url http://127.0.0.1:8899/v1 \
  --session-id "$REPLAY_ID" \
  --extra-header "x-conversation-id=$REPLAY_ID"
```

식별 헤더를 보호 기능 설정에 맞추고, 비교할 때마다 새 ID를 쓰며, 인증이 필요하면 `--api-key-env` 를 추가합니다.
> Match the identity headers to your protection configuration, use a fresh ID for
> each comparison, and add `--api-key-env` when authentication is required.

<details markdown="1">
<summary>이 재실행이 측정하는 것과 측정하지 않는 것 / What this replay measures, and what it does not</summary>

픽스처가 기록을 제공합니다. 생성된 답은 이후 요청에 들어가지 않으며, 도구도 실행되지 않습니다. 이 재실행은 요청의 형태와 라우팅 진단 정보를 측정하며, **완료된 에이전트 작업의 품질은 측정하지 않습니다.**
> The fixture supplies the history: generated answers are not fed into later
> requests, and tools are not executed. It measures request shape and routing
> diagnostics, **not completed-agent-task quality**.

현재 CLI는 응답 본문이 잘못된 형식이어도 HTTP 2xx를 성공으로 셉니다. 또한 도구 루프 전환 위반을 보고하면서도 종료 코드 0으로 끝날 수 있습니다. 모든 턴의 모델 식별 정보를 포함해 보고서를 확인하세요. 드라이 런(dry run)에서 전환 횟수가 0이라는 것은 연속성의 근거가 아닙니다. 캐시된 토큰 정보가 없으면 0으로 기록되므로, 데이터가 없는 경우와 캐시 재사용이 없는 경우를 구분하려면 백엔드 텔레메트리를 사용합니다.
> The current CLI counts HTTP 2xx as success, even for a malformed response body,
> and can exit zero while reporting a tool-loop switch violation. Check the
> report, including model identity on every turn. Dry-run zero switch counts
> are not continuity evidence. Missing cached-token detail is recorded as zero;
> use backend telemetry to distinguish missing data from no cache reuse.

</details>

생성된 답이 다음 동작을 이끄는 작업에는 [sr-bench의 고정 계획(frozen-plan) 워크플로우](https://vllm-sr.ai/docs/benchmarking/sr-bench/plan-and-run)와 [작업 보고서](https://vllm-sr.ai/docs/benchmarking/sr-bench/results)를 사용합니다. 기본 에이전트 실행은 세션 식별자를 제공하지 않으며, [작업별 식별자는 별도로 추적 중입니다](https://github.com/vllm-project/semantic-router/issues/4256). 이 기본값으로 실행한 결과를 세션 범위 보호(session-scoped protection)에 대한 테스트로 여기지 마세요.
> For tasks where generated answers drive subsequent actions, use
> [sr-bench's frozen-plan workflow](https://vllm-sr.ai/docs/benchmarking/sr-bench/plan-and-run) and
> [task reports](https://vllm-sr.ai/docs/benchmarking/sr-bench/results). Default agent runs do not
> supply session identity; [per-task identity is tracked separately](https://github.com/vllm-project/semantic-router/issues/4256).
> Do not treat those defaults as tests of session-scoped protection.

### 참고 자료와 재현성 / References and reproducibility

- [연속성 보호](https://vllm-sr.ai/docs/tutorials/learning/protection), [메모리와 Replay](https://vllm-sr.ai/docs/tutorials/learning/memory-and-replay), [최근 결과 게이트](https://github.com/vllm-project/semantic-router/pull/3436).
- [도구 루프 소유권 수정](https://github.com/vllm-project/semantic-router/pull/4297), [Chat/Responses 벤치마크 실행기](https://github.com/vllm-project/semantic-router/pull/4082), [코딩 에이전트 재실행](https://github.com/vllm-project/semantic-router/pull/4154), [작업별 연속성 보고](https://github.com/vllm-project/semantic-router/pull/4239).
- [vLLM 프리픽스 캐싱](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)과 [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/).
- [평가 기록과 홀드아웃 결과](https://github.com/vllm-project/semantic-router/issues/4080#issuecomment-5938865660). 라우터 리비전: [`0ee9955fc6ae0d79104e33bc933ae0f75ed52acf`](https://github.com/vllm-project/semantic-router/commit/0ee9955fc6ae0d79104e33bc933ae0f75ed52acf), Qwen3-0.6B: [`c1899de289a04d12100db370d81485cdf75e47ca`](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), Qwen3-8B: [`b968826d9c46dd6066d109eabc6255188de91218`](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218).

> - [Continuity protection](https://vllm-sr.ai/docs/tutorials/learning/protection),
>   [memory and Replay](https://vllm-sr.ai/docs/tutorials/learning/memory-and-replay), and
>   [recent-outcome gates](https://github.com/vllm-project/semantic-router/pull/3436).
> - [Tool-loop ownership fix](https://github.com/vllm-project/semantic-router/pull/4297),
>   [Chat/Responses benchmark runner](https://github.com/vllm-project/semantic-router/pull/4082),
>   [coding-agent replay](https://github.com/vllm-project/semantic-router/pull/4154), and
>   [per-task continuity reporting](https://github.com/vllm-project/semantic-router/pull/4239).
> - [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
>   and [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/).
> - [Evaluation record and holdout results](https://github.com/vllm-project/semantic-router/issues/4080#issuecomment-5938865660).
>   Router revision: [`0ee9955fc6ae0d79104e33bc933ae0f75ed52acf`](https://github.com/vllm-project/semantic-router/commit/0ee9955fc6ae0d79104e33bc933ae0f75ed52acf);
>   Qwen3-0.6B: [`c1899de289a04d12100db370d81485cdf75e47ca`](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca);
>   Qwen3-8B: [`b968826d9c46dd6066d109eabc6255188de91218`](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218).
