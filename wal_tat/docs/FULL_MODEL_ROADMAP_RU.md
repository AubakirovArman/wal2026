# Маршрут WAL-TAT до полной ternary-модели

## Определение финиша

Полная модель считается готовой, когда:

1. все 28 decoder blocks и tied embedding/head имеют зафиксированный формат;
2. независимые PPL и task-benchmarks проходят общий, заранее заданный budget;
3. checkpoint воспроизводим из исходной BF16-модели и WAL;
4. codes реально упакованы, а runtime не материализует всю матрицу в BF16;
5. измерены file size, peak RAM/VRAM, prefill и decode speed.

Основная исследовательская ветка: ternary `{-1,0,+1}`, logical 1.585 bit и
physical Q2-g128 2.125 bpw. Практическая rate--distortion ветка разрешает
signed Q4-g128 4.125 bpw для групп, которые не проходят sealed generalization.
Последний защитный формат — signed Q8-g128 8.125 bpw. Форматы считаются
раздельно; Q4/Q8 никогда не записываются в ternary coverage.

## Текущий счётчик

```text
strict decoder blocks: 1 / 28 complete
mixed low-bit blocks:  6 / 28 complete, 22 remain
strict major matrices: 11 / 197 complete
mixed major matrices:  42 / 197 complete, 155 remain
strict ternary weights: 81,250,048 / 1,720,451,072 = 4.722601% in mixed artifact
Q4 rescue weights:      75,278,080 / 1,720,451,072 = 4.375485%
Q8 rescue weights:      145,461,760 / 1,720,451,072 = 8.454862%
all low-bit weights:    301,989,888 / 1,720,451,072 = 17.552948%
remaining high precision: 1,418,461,184 = 82.447052%
embedding/head:       0 / 1 tied matrix
packed runtime:       Q2 reference packer ready; mixed Q2/Q4/Q8 packer and kernels remain
```

## Система quality budgets

`1.02` остаётся только диагностическим micro-gate. Он появился как наша
эвристика «не принимать локальный шаг хуже teacher более чем на 2% NLL» и не
является внешним стандартом.

| Уровень | Данные | Назначение |
|---|---|---|
| Micro transaction | маленькие frozen suites | быстрый rollback, limit 1.02 |
| Block development | recovery domains | выбор optimizer/objective |
| Sealed block audit | непересекающиеся, ранее не раскрытые domains | проверка generalization |
| Cumulative model audit | один неизменяемый большой harness | общий NLL/PPL budget |
| Task audit | reasoning/code/instruction/knowledge | поведенческое качество |
| Runtime audit | packed artifact | память и скорость |

Для эксперимента задаются несколько итоговых кривых: например `+3%`, `+5%` и
`+10% NLL` всей модели. Это проектные цели, а не числа из внешней статьи.
Пропорциональная guide-линия для coverage `c`:

```text
allowed_ratio(c, B) = 1 + B*c
```

Она не считается физическим законом: block errors взаимодействуют нелинейно и
могут быть восстановлены end-to-end QAT. Это консервативный индикатор, который
не позволяет ошибочно назвать успешным рецепт, расходующий весь budget на
первых blocks.

Если цель сформулирована как `+5% PPL`, формула другая, поскольку
`PPL=exp(NLL)`:

```text
allowed_NLL_ratio(domain) = 1 + c*log(1.05)/teacher_NLL(domain)
```

Поэтому в результатах всегда явно указывается, ограничивается NLL или PPL.

## Выполненные фазы

### Фаза 0 — транзакционный механизм

- g128 ternary codes и scales;
- causal activation/Fisher ranking;
- exact commit/rollback;
- hash-chained WAL;
- multi-domain NLL gate;
- matched controls.

### Фаза 1 — MLP compensation

- `down_proj`, `up_proj`, `gate_proj` layer 27 доведены до 100%;
- triadic SwiGLU window связывает gate/up/down;
- scale-only compensation проходила там, где candidate-only откатывалась;
- adaptive transaction sizes позволили пройти локальные frontiers.

### Фаза 2 — attention и полный block 27

- GQA-aware V/O окна конвертировали `v_proj` и `o_proj`;
- GQA-aware K/Q окна конвертировали `k_proj` и `q_proj`;
- все семь крупных матриц layer 27 имеют только ternary codes;
- diverse fixed-code recovery улучшил независимый audit с ~1.05 до максимум
  1.01404.

### Фаза 3 — proxy-code recovery и sensitivity map

- hard forward остаётся точным `{-scale, 0, +scale}`;
- гладкая proxy-переменная используется только для backward;
- layer 27 после proxy recovery проходит два audit suite с ratios ниже `1`;
- два независимых sensitivity scan дали одинаковый порядок blocks
  (Spearman `1.0`);
- следующим выбран наименее чувствительный remaining `layer 24`.

## Фаза 4 — block 24: strict frontier и принятый mixed fallback

Strict checkpoint `s0053` достиг 76,673,024 ternary weights (`4.456565%`).
Попытка одномоментно перевести оставшиеся 23,990,272 MLP-веса показала, что
compute не является узким местом: полный macro-run занимает минуты, но
activation-aware hard ternary даёт code ratio около `1.03985`. Scale-only,
logit-KD, global proxy flips и first-order hard repair (включая один flip и
code-only gradient) не прошли development и были откатаны.

Rate--distortion ablation затем оставил ранее принятые группы Q2, а самые
трудные группы представил signed Q4-g128. Минимальный проверенный passing
вариант использует 178,053 Q4-группы и 9,371 новых ternary-групп. Он даёт
77,872,512 strict-ternary и 22,790,784 Q4 weights, полностью закрывает второй
decoder block и имеет `3.332499 bpw` по трём MLP-матрицам.

Fresh reload воспроизвёл development ratios
`0.994024 / 0.999487 / 1.018691`. Sealed audit-v26 прошёл absolute gate
`0.992737 / 1.002540 / 0.955660` и incremental gate против `s0053`
`1.001073 / 1.002089 / 1.002177 <= 1.005`. Audit-v26 теперь раскрыт.

## Фаза 5 — composable mixed compiler и block 23

Accepted Q2/Q4 artifact стал composable parent: loader разрешает новые
матрицы с нулевой source-mask, но побитно защищает все прежние Q2 codes/scales.
Полный Q4 layer 23 прошёл incremental gate, но code ratio `1.021478` не прошёл
absolute `1.02`. Fixed-code Q4 scale-QAT ухудшил code до `1.021982` и был
отклонён. Projection ablation локализовал ущерб в `gate/up`.

Минимальный проверенный passing вариант оставил `62.5%` групп блока Q4 и
перевёл `37.5%` групп в Q8. Средний budget layer 23 равен `5.625 bpw`.
Fresh reload дал `0.994688 / 1.000512 / 1.019039`; one-shot audit-v27 —
`0.991963 / 1.005302 / 0.959639`, incremental worst `1.003018 <= 1.005`.
Это завершило третий low-bit decoder block без BF16 в его крупных матрицах.

Первый rate--distortion reverse-compression шаг layer 23 затем перевёл ещё
`597,632` Q4-веса в строгий Q2. Теперь блок содержит `1.187388%` Q2,
`61.312612%` Q4 и `37.5%` Q8 при `5.601252 bpw`. Fresh reload дал
`0.994884 / 1.001022 / 1.019857`; prospectively объявленный audit-v30 —
`0.992754 / 0.992227 / 0.961534`, incremental worst против immutable strict
source `1.003856 <= 1.005`. Новый accepted artifact имеет SHA-256
`f7394ff69e348c89ba424369c004c719df05c55e8d23315687758f3b7d573eeb`.

Следующим по двум совпавшим sensitivity scan был обработан layer 25. Full-Q4
candidate не прошёл официальный absolute code gate, поэтому Q8 allocator
нашёл минимальную full-development точку `7.999929%` Q4 и `92.000071%` Q8
при `7.805003 bpw`; точки до `91.9%` Q8 были отклонены без ослабления порога.
Fresh reload дал `0.994942 / 1.001131 / 1.019953`. Prospective one-shot
audit-v31 прошёл с absolute ratios `0.993532 / 1.003477 / 0.971895` и
incremental worst `1.004159 <= 1.005`. Пятый полный low-bit decoder block
увеличил coverage до `14.627457%`; accepted artifact SHA-256:
`466fb84e60f851e938a249db07ac38bf22e1ff4bcaa4eda68ef5f1aed6b1ffa2`.

Первый layer-25 Q8→Q4 rate--distortion шаг затем выбрал 47,029 наименее
чувствительных Q8-групп: 6,019,712 весов стали Q4. Доля Q4 блока выросла до
`19.960022%`, Q8 снизилась до `80.039978%`, а budget — с `7.805003` до
`7.326599 bpw`. Новый WikiText-103/C4/datasets-code audit-v32 прошёл с
absolute ratios `0.993773 / 0.943618 / 0.968501`; incremental worst равен
`1.003159` против strict source и `1.000108` против принятого parent.
Accepted artifact SHA-256:
`6d09ee3f4a697467a21f3d5da6987a5ec413cb4c4ff3eb345d1eff6b18291dc8`.

Второй layer-25 Q8→Q4 шаг уточнил полный development frontier между `0.2%`
и `0.5%` оставшихся Q8-групп. Максимум составил 787 групп, или 100,736
весов; следующий размер уже не прошёл code gate. После принятия layer 25
содержит `20.160166%` Q4 и `79.839834%` Q8 при `7.318593 bpw`. Sealed
audit-v33 дал absolute ratios `0.991459 / 0.940683 / 0.942708`, incremental
worst `1.003351` против strict source и `1.000018` против parent. Новый
accepted artifact SHA-256:
`6f6da5385d37f77773af9c1c15a317282cdf93dfe31c1ee2cf84b4eb2f48cf2c`.

Первый layer-25 Q4→Q2 шаг затем использовал тот же activation-weighted
rate--distortion selector. Full development принял `13.8%` Q4-групп, а
`13.9%` уже не прошло selection code gate. В strict ternary переведены
10,940 групп, или 1,400,320 весов. Теперь блок содержит `2.782186%` Q2,
`17.377981%` Q4 и `79.839834%` Q8 при `7.262950 bpw`. Audit-v34 дал
absolute ratios `0.993359 / 0.943748 / 0.961192`; incremental worst равен
`1.004102` против strict source и `1.000934` против parent. Новый accepted
artifact SHA-256:
`bf17ca58ad836ef1f0569b6ae9d2d28f7224cfb383a64898dfbaef2f9acf5701`.

Следующий глобальный поиск показал, что новый `1%` Q4-хвоста layer 25 уже не
проходит absolute code gate. Полный `7→5→3` collapse его 8,746,624
Q4-весов также отклонён: все стадии завершились с нулевым hard-code churn, а
финальные ratios составили `1.002569 / 1.013937 / 1.034209`. Поэтому selector
перешёл к Q4-группам `layer 24 MLP`. Там максимум full-development составил
1,781 группу, или `227,968` весов; `2%` уже не прошли selection code gate.
Новый кандидат прошёл fresh reload и sealed audit-v35 с absolute ratios
`0.992240 / 0.944613 / 0.966782`, incremental worst `1.004161` против strict
source и `1.000162` против parent. Весь layer 24 теперь содержит
`55.171712%` Q2 и `44.828288%` Q4 при `3.021566 bpw`. Accepted artifact:
`b4938a3417fa59bc89075dc1af4edd509c2e44c1bfe6ff9135729222a5a6746d`.

После этого progressive collapse был изменён с full-tail на masked
rate--distortion режим. Из 68,333 оставшихся Q4-групп layer 25 заранее
выбраны 683 группы с минимальной конечной ternary distortion; только они
прошли hard codebooks `7→5→3`, а прочие Q4, все Q8 и прежние Q2 остались
побитно неизменными. Новый шаг перевёл `87,424` веса в strict Q2. Layer 25
теперь содержит `2.955882%` Q2, `17.204285%` Q4 и `79.839834%` Q8 при
`7.259476 bpw`. Fresh reload дал `0.995253 / 1.001842 / 1.019936`, а
one-shot audit-v36 — `0.992109 / 0.943986 / 0.957007`; incremental worst
равен `1.003765` против strict source и `1.000022` против accepted parent.
Новый accepted artifact SHA-256:
`01a3176b1971c76868bcd55de068ba5a3058278ad849df4ee7e496a9b98780f4`.

Непосредственно следующий `1%` layer 25 не прошёл code gate (`1.020077`),
а уменьшение до `0.5%` не помогло (`1.020184`). Masked `1%` layer-24 MLP
также отклонён при `1.020546`. Вместо дальнейшего дробления атомов построена
новая 75%-code calibration с отдельными recurring-v8 gates. Proxy-recovery
почти прошёл заранее заданный improvement, но после step 288 стал дискретно
нестабилен. Финальный scale-only arm заморозил все Q4 codes и выбрал step 320.

Полученный coverage-neutral artifact изменяет ровно `105,060` FP16 Q4-scale
layer-24 MLP и сохраняет побитно все codes, masks, Q2 и Q8. Fresh v10 дал
`0.991672 / 0.997713 / 0.988142`; one-shot audit-v37 —
`0.993252 / 0.938675 / 0.935348`. Incremental worst равен `1.003535` против
strict source и `1.000078` против accepted parent. Coverage и projected bpw
не изменились. Новый quality-recovered parent SHA-256:
`f5f50147edcf6bf912fbfb476dc8c50ca494316bd1d52268f55982b6c900f0c9`.

После этого тот же второй masked `1%` layer-25, ранее отклонённый на
исчерпанном parent, прошёл. Из оставшихся Q4 выбраны 676 минимально
чувствительных групп; `7→5→3` перевёл `86,528` весов в strict Q2. Fresh
reload дал `0.995292 / 1.001841 / 1.019975`, one-shot audit-v38 —
`0.997427 / 0.940653 / 0.964373`; incremental worst равен `1.004710` против
strict source и `1.000047` против recovered parent. Layer 25 теперь содержит
`3.127797%` Q2, `17.032369%` Q4 и `79.839834%` Q8 при `7.256037 bpw`.
Accepted artifact SHA-256:
`411c8d337405c8f2e2f48288ffa33a9b7d4ca647c63fd611304be0ec5c054588`.

Scale-only recovery Q4-групп самого layer 25 не набрал заранее заданный
improvement и был отклонён. Повторный joint scale-only arm для layer-24 MLP
изменил `35,378` FP16 scales при полностью неизменных codes и masks. Fresh
v10 дал `0.991688 / 0.997828 / 0.988282`; one-shot audit-v39 —
`0.992793 / 0.940519 / 0.964751`. Incremental worst равен `1.004426` против
strict source и `1.000003` против parent. Coverage-neutral accepted artifact:
`5f35a9e7ac28138b358bb22762563faaf037a00d8bc52826712c6ce90d3f0ea7`.

Следующий layer-25 атом уменьшен до `0.25%`: 167 групп (`21,376` весов)
прошли `7→5→3` и стали strict Q2. Fresh v8 дал
`0.995280 / 1.001836 / 1.019925`; one-shot audit-v40 —
`0.997258 / 0.938989 / 0.974302`. Incremental worst равен `1.003959` против
strict source и `1.000052` против recovered parent. Layer 25 теперь имеет
`3.170268%` Q2, `16.989899%` Q4 и `79.839834%` Q8 при `7.255188 bpw`.
Accepted artifact:
`7494fa959b20a089ca7d7bb7afc48f15b67c9b7f6bdb88a5f3c9937dd77e0bd6`.

Следующим завершён layer 26. Full-Q4 и два постепенно расширенных Q8 rescue
прошли development, но audit-v41 и audit-v42 выявили code-domain source gaps
`1.005861` и `1.006725` при фиксированном лимите `1.005`; оба кандидата были
отклонены. Финальная раскладка оставляет `o_proj` в Q4, а остальные шесть
матриц — в Q8 (`7.791667 bpw`). Добавленный Q8-scale recovery на code-heavy
calibration изменил только 12,219 Q4-scale и 139,271 Q8-scale, сохранив все
codes и masks. Fresh v10 прошёл, а новый audit-v43 на C4, WikiText-103 и
sklearn code дал `0.992036 / 0.940191 / 0.978863`; incremental worst равен
`1.003721` против strict source и `1.000117` против parent. Это шестой полный
low-bit decoder block. Accepted artifact SHA-256:
`148908dbc2f43d72d73402904e68a6642c8d1e1817c6400cd4209d0b30f7c76a`.

Layer 22 показал распределённую чувствительность: полный Q4 имел development
code ratio `1.021613`. Минимальная проверенная глобальная rescue-точка оставила
`19.999949%` групп Q4 и перевела `80.000051%` в Q8 (`7.325002 bpw`). Fresh
reload дал `0.994716 / 1.000847 / 1.019173`; audit-v28 —
`0.991651 / 0.993830 / 0.938898`, incremental worst `1.002913`. Это четвёртый
полный low-bit block, но одновременно сигнал остановить слепое накопление Q8 и
добавить reverse-distillation Q8→Q4→Q2.

Первый reverse-distillation audit показал границу прямого Q8-teacher→Q4
восстановления. Scale-only recovery не улучшил full-Q4 candidate. При
hard-forward Q4 proxy с `lr=0.005` codes практически не пересекали ячейки; с
`lr=0.02` на лучшем шаге 64 изменилось около `0.05–0.19%` codes, но code ratio
остался `1.021875 > 1.02`. При `4–5.5%` churn на шаге 128 качество резко
ухудшилось. Оба кандидата отклонены, accepted artifact не изменён. Это
обосновывает staged collapse с отдельным восстановлением и freeze на каждой
ступени, а не дальнейший brute-force подбор proxy learning rate.

Полный staged collapse `Q4→7→5→3` также не прошёл: на финальном ternary
уровне development ratios достигли `1.010909 / 1.025808 / 1.065499`.
Однако этот отрицательный результат подтвердил более сильную
rate--distortion постановку: не схлопывать все Q4-группы одновременно, а
ранжировать их по activation-weighted ternary error и переводить только
безопасную долю. Первый такой reverse-compression шаг заменил `956,288`
весов Q4 на строгий Q2. Теперь layer 22 содержит `1.899974%` Q2,
`18.099976%` Q4 и `80.000051%` Q8 при `7.287003 bpw`.

Fresh reload нового artifact дал `0.994856 / 1.000954 / 1.019816`.
Prospectively объявленный one-shot audit-v29 прошёл с absolute ratios
`0.991622 / 0.989226 / 0.973866` и incremental ratios против immutable
strict source `1.001776 / 1.002720 / 1.004860 <= 1.005`. Accepted artifact:
`wal-tat-block22_reverse_q4_to_q2_fraction_refine_v2-mixed-q2-q4-q8.pt`,
SHA-256 `b16392f87c84d5facbbef3dec195c142a45187ffdb7ef652090fac4c5bf433db`.
Общий low-bit coverage не изменился (`11.701966%`), но strict ternary coverage
выросла до `4.581868%`, а projected payload уменьшился на `0.228 MiB`.

Следующий этап — versioned packed Q2/Q4/Q8 layout и перенос composable compiler
на следующие блоки. Параллельно strict research продолжает progressive
`Q4 -> 7 -> 5 -> 3`, transform-space и joint codebook recovery. Для этой ветки
добавлен общий activation-weighted projector нечётных symmetric codebooks;
первый end-to-end staged run остаётся checkpoint-neutral до прохождения gate.

Отдельная family-transfer гипотеза: сравнить одинаковый ternary recipe на
обычном BF16, Q4-QAT-unquantized master checkpoint и деквантизированном Q4.
Она проверит, является ли Q4-QAT basin более удобным low-bit manifold. Этот
Gemma-эксперимент не заменяет и не задерживает текущую Qwen campaign.

### Историческая strict lineage block 24

Q/K/V/O layer 24 приняты и совместно с layer 27 прошли point gate на
recurring validation-v3/v4. Через sensitivity-ranked транзакции также приняты
75.78125% `up_proj`, 10.611979% `gate_proj` и 14.615885% `down_proj`.
Текущее покрытие `4.395617%`, условная `+5% NLL` guide равна `1.00219781`,
худший измеренный точечный ratio равен `1.001472`. Кумулятивный paired SQuAD
upper-95 равен `1.006937` на v3 и `1.003597` на v4, но последний D10-шаг был
принят уже по prospective dual gate: cumulative point плюс incremental
upper-95 `<=1.0002`. Его incremental maxima равны `1.00005088` и
`1.00010712`; это не заменяет будущий sealed block audit.

Lineage checkpoint `s0048r3` сохранил то же coverage, но после
target-local fallback-compensation QAT проходит новый audit-v8 с абсолютными
C4/SQuAD/code ratios `0.993307 / 0.992380 / 0.985639`; incremental paired
upper-95 относительно parent равны `0.998246 / 0.996783 / 0.994289`. Это
создаёт quality headroom для следующего роста, но audit-v8 уже раскрыт и не
будет использоваться для подбора следующего кандидата. До следующей
тернаризации строится заранее зафиксированный audit-v9.

После этого audit-v9 был заморожен и проверен на отсутствие пересечений с 18
retained suites. Development-only scan выбрал 96 D1-групп `gate_proj` с
`threshold_ls`; proxy-QAT не превзошёл step zero и потому не изменил codes.
Один заранее объявленный audit-v9 дал incremental upper-95
`1.000039 / 1.000028 / 1.000000`, кандидат принят как `s0049`. Добавлено
12,288 strict-ternary weights, coverage `gate_proj` вырос до `6.4453125%`, а
общий счётчик — до `74,575,872`.

До следующего шага был построен audit-v10 на трёх новых диапазонах; overlap
checker доказал нулевое пересечение со всеми 19 retained suites. Prospective
scan увеличил атом до 2,048 g128-групп (262,144 weights), то есть в 21.33 раза.
Победил `gate_proj + activation_wls`; recovery выбрал шаг 128 при нулевом
churn кодов. One-shot audit-v10 дал incremental upper-95
`1.000039 / 1.000336 / 1.000148`, все ниже заранее масштабированного по
`sqrt(2048/96)` лимита `1.000924`. Кандидат принят как `s0050`, coverage
`gate_proj` вырос до `8.528646%`, а общий счётчик — до `74,838,016`.

Audit-v11 отклонил следующий 262,144-weight кандидат и выявил переносимый
SQuAD-train gap самого `s0050`. Coverage-neutral recovery изменил только
непринятые BF16-группы layer-24 MLP и сохранил все ternary codes/scales/masks.
На one-shot audit-v12 он улучшил frontier на всех доменах: incremental
upper-95 `0.998983 / 0.998855 / 0.997520`. Опубликованный `s0050r1`
fresh-verifies at `wiki=0.946518`, `code=0.957625`; coverage остаётся
`74,838,016`. Перед новым coverage candidate нужен audit-v13, поскольку v12
теперь закрыт после одного использования.

Audit-v13 был заморожен на новых диапазонах до выбора кандидата и показал
нулевое пересечение с 24 retained suites. Новый scan снова выбрал атом из
2,048 D1-групп `gate_proj + activation_wls`; proxy recovery сохранил code
churn равным нулю. На единственном audit-v13 худший cumulative point ratio
равен `1.000515`, а худший incremental upper-95 — `1.000245`, ниже заранее
заданного лимита `1.000924`. Atomic checkpoint `s0051` добавил 262,144 веса,
поднял общий счётчик до `75,100,160` (`4.365144%`) и fresh-verifies at
`wiki=0.946587`, `code=0.957546`. Audit-v13 закрыт; следующий кандидат требует
нового audit-v14.

Попытка ускорить следующий шаг до 8,192 D1-групп `down_proj` прошла
development, но audit-v14 отклонил её по code incremental upper-95:
`1.001969 > 1.001848`. Новый audit-v15 был заморожен до уменьшения атома.
Фиксированный 4,096-group retry уже прошёл incremental gate с максимумом
`1.001094 <= 1.001306`, но выявил pre-existing SQuAD ratio `1.004861` у
frontier `s0051`; cumulative candidate ratio `1.005260` выше guide
`1.002198`. Оба кандидата отклонены без изменения coverage. Перед дальнейшим
ростом требуется coverage-neutral generalization recovery на свежих данных и
новый sealed audit-v16.

Две последовательные coverage-neutral recovery стадии на новых
непересекающихся suites сформировали `s0051r2`. После этого ранее отклонённый
4,096-group `down_proj` atom прошёл one-shot audit-v19: cumulative worst point
ratio `0.995173`, incremental upper-95 worst `1.000710 <= 1.001306`.
Опубликованный `s0052` добавил 524,288 strict-ternary weights, поднял
`down_proj` до `14.615885%`, а общий coverage — до `75,624,448`
(`4.395617%`).

Новая 8,192-group попытка прошла incremental audit-v20, но cumulative SQuAD
ratio `1.005280` превысил guide `1.002198`; coverage не изменился. После
двухступенчатого late-range recovery второй, half-rate этап прошёл новый
audit-v22: C4/SQuAD/NumPy-code point ratios равны
`0.991022 / 1.001470 / 0.972487`, incremental upper-95 worst равен
`0.998832 <= 1.0005`. Текущий `s0052r1` сохранил все ternary codes и coverage,
а fresh reload дал Wiki/Code relative NLL `0.943734 / 0.953158`. Следующая
транзакция использует заранее построенные development-v8 и audit-v23.

Masked proxy recovery теперь поддерживает этот частичный block без ложной
тернаризации оставшихся BF16-групп. На текущем frontier он сохранил coverage и
uncommitted master weights, сменил только два уже принятых ternary-кода и
перенастроил их shared scales. Независимый worst ratio снизился с
`1.001531213` до `1.000654804`; при том же gate `1.00195057` доступный
headroom вырос более чем втрое. Это позволяет снова испытывать более крупные
атомы вместо бесконечной последовательности `1/1024`.

Практическая проверка подтвердила пользу recovery: `+6.25% up_proj` лишь
немного не прошёл неизменённый development gate, а автоматически уменьшенный
`+3.125%` атом прошёл development, fresh reload и оба holdout. Он добавил
393,216 ternary weights одним commit и поднял `up_proj` до `35.64453125%`;
лучшим оказался linked BF16 `down/gate` arm с worst ratio `1.000731191`.
Повторный атом того же размера также прошёл оба holdout, добавил ещё 393,216
weights и поднял `up_proj` до `38.76953125%`; linked arm снова выиграл с worst
ratio `1.000755065`. Третий атом прошёл, поднял coverage до `41.89453125%`,
но candidate-only впервые обошёл linked arm на holdout (`1.001118535` против
`1.001146107`), поэтому BF16-компенсация не закрепляется без проверки.
Следующий masked proxy recovery сохранил все discrete codes и все
uncommitted BF16/scale элементы, изменил 467,770 committed scales и улучшил
audit-v3 worst до `1.000297213` без изменения coverage.
Из этого frontier принят `down_proj +0.390625%`: 49,152 новых ternary weights,
оба holdout пройдены, linked arm выиграл с worst ratio `1.000391478`, а
`down_proj` достиг `1.07421875%`.
Затем принят `gate_proj +0.1953125%`: 24,576 новых weights, оба holdout
пройдены, linked arm выиграл с worst `1.000349834`, coverage gate достиг
`0.390625%`.
Следующий round-robin атом `up_proj +3.125%` добавил 393,216 weights и прошёл
оба holdout; linked arm выиграл с worst `1.000441498`, coverage up достиг
`45.01953125%`.
Следующий down-атом добавил 49,152 weights, прошёл оба holdout и поднял
`down_proj` до `1.46484375%`; linked arm worst равен `1.000459295`.
Следующий gate-атом добавил 24,576 weights и поднял `gate_proj` до
`0.5859375%`. Оба arm-а прошли два holdout; candidate-only выиграл с worst
`1.000469347` против `1.000483896` у linked BF16-окна.
Следующий up-атом добавил 393,216 weights и поднял `up_proj` до
`48.14453125%`. Linked arm выиграл с worst `1.000691962` против
`1.000778586` у candidate-only.
Следующий down-атом добавил 49,152 weights и поднял `down_proj` до
`1.85546875%`. Linked arm выиграл с worst `1.000677098` против
`1.000722296` у candidate-only.
Следующий gate-атом добавил 24,576 weights и поднял `gate_proj` до
`0.78125%`. Linked arm выиграл с worst `1.000642781` против `1.000709085` у
candidate-only.
Следующий up-атом добавил 393,216 weights и поднял `up_proj` до
`51.26953125%`. Linked arm выиграл с worst `1.000961833` против
`1.000986563`; normalized headroom сузился до `0.525152`.
Следующий down-атом добавил 49,152 weights и поднял `down_proj` до
`2.24609375%`. Оба arm-а прошли оба holdout; linked BF16 MLP arm едва
выиграл по независимому worst ratio (`1.001010915` против `1.001012296`).
Normalized headroom равен `0.501272`: следующим выполняется один gate-атом,
затем masked proxy recovery перед новым крупным up-ростом.
Этот gate-атом добавил 24,576 weights и поднял `gate_proj` до `0.9765625%`.
Linked arm выиграл holdout (`1.001018946` против `1.001023001`), normalized
headroom равен `0.497488`. Следующий шаг — masked proxy recovery без изменения
coverage, затем новый cumulative audit.
Recovery прошёл оба cumulative holdout и улучшил worst ratio до
`1.000115086`, сохранив masks, coverage и все непринятые BF16 master weights.
Изменился один активный ternary-код и 476,026 committed scales; 495 изменений
codes под нулевыми masks не участвуют в forward. Normalized headroom вырос до
`0.943243`, поэтому следующий безопасный опыт — `up_proj +3.125%`.
Этот опыт прошёл оба holdout и добавил 393,216 weights: `up_proj` достиг
`54.39453125%`. Candidate-only выиграл у linked arm (`1.000404500` против
`1.000427594`), normalized headroom остался `0.801631`. Следующий шаг
round-robin — `down_proj +0.390625%`.
Этот down-атом также прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `2.63671875%`. Linked arm выиграл (`1.000412414` против
`1.000453858`), normalized headroom равен `0.797892`. Следующий round-robin
атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.3671875%`. Candidate-only выиграл (`1.000953523` против
`1.000995915`), normalized headroom равен `0.535963`. Следующая попытка —
`up_proj +3.125%`; при отказе или узком запасе выполняется recovery.
Попытка не прошла development gate: candidate-only SQuAD ratio равен
`1.002592097`, linked — `1.002554625`, gate — `1.002066271`. Coverage и
принятый checkpoint не изменились, следующий up-размер автоматически уменьшен
до `1.5625%`. Перед повтором выполняется masked proxy recovery.
Recovery v4 сохранил coverage, все codes и непринятые BF16 master weights,
изменив 481,135 committed scales. Оба cumulative holdout пройдены; worst
улучшился с `1.000953523` до `1.000301811`, normalized headroom вырос до
`0.853122`. Следующая up-попытка использует уменьшенную долю `1.5625%`.
Эта уменьшенная транзакция прошла оба holdout, добавила 196,608 weights и
подняла `up_proj` до `59.08203125%`. Linked arm минимально выиграл у
candidate-only (`1.000243471` против `1.000255685`), normalized headroom
равен `0.881842`. Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `3.41796875%`. Candidate-only выиграл у linked arm по
независимому worst (`1.000307810` против `1.000332156`), normalized headroom
равен `0.850722`. Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.5625%`. Linked arm выиграл у candidate-only по независимому
worst (`1.000336331` против `1.000357894`), normalized headroom равен
`0.836946`. Следующий round-robin атом — уменьшенный `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `60.64453125%`. Candidate-only выиграл у linked arm по независимому worst
(`1.000550819` против `1.000677756`), normalized headroom равен `0.733700`.
Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `5.37109375%`. Candidate-only выиграл у linked arm
(`1.000605718` против `1.000739123`), normalized headroom равен `0.711737`.
Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом тоже прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.5390625%`. Candidate-only выиграл у linked arm
(`1.000627927` против `1.000815486`), normalized headroom равен `0.701269`.
Следующий round-robin атом — `up_proj +1.5625%` без предварительного recovery.
Попытка `up_proj +1.5625%` была откатана на development-gate. Уменьшенный
атом `+0.78125%` прошёл оба holdout, добавил 98,304 weights и поднял
`up_proj` до `67.67578125%`. Holdout выбрал linked arm вместо candidate-only
(`1.000638609` против `1.000662713`), normalized headroom равен `0.696599`.
Следующий round-robin атом — `down_proj +0.390625%`.
Следующий down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `5.76171875%`. Независимый аудит выбрал candidate-only вместо
development-победителя linked arm (`1.000809223` против `1.000834380`),
normalized headroom равен `0.615802`. Следующий round-robin атом —
`gate_proj +0.1953125%`.
Следующий gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.734375%`. Candidate-only выиграл у linked arm
(`1.000730373` против `1.000794258`), normalized headroom равен `0.653356`.
Следующий контролируемый атом — `up_proj +0.78125%`.
Этот up-атом прошёл оба holdout, добавил 98,304 weights и поднял `up_proj`
до `68.45703125%`. Candidate-only выиграл у linked arm (`1.000762124` против
`1.000833618`), normalized headroom равен `0.638776`. Следующий round-robin
атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `6.15234375%`. Candidate-only выиграл у linked arm
(`1.000850378` против `1.000987898`), normalized headroom равен `0.597219`.
Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.9296875%`. Holdout минимально предпочёл linked arm
вместо candidate-only (`1.000883711` против `1.000884797`), normalized headroom
равен `0.581573`. Следующий контролируемый атом — `up_proj +0.78125%`.
Этот up-атом прошёл оба holdout, добавил 98,304 weights и поднял `up_proj`
до `69.23828125%`. Candidate-only выиграл у linked arm (`1.001048501` против
`1.001158013`), normalized headroom равен `0.504217`. Следующий round-robin
атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `6.54296875%`. Holdout выбрал candidate-only вместо
development-победителя linked arm (`1.001052724` против `1.001137076`),
normalized headroom равен `0.502556`. Следующий атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `3.125%`. Candidate-only выиграл у linked arm
(`1.001044083` против `1.001077155`), normalized headroom немного вырос до
`0.506806`. Перед следующим up-атомом выполняется coverage-neutral recovery.
Masked proxy recovery v6 сохранил coverage, masks и все непринятые BF16
master weights. Изменился один активный ternary-код в `o_proj`, 494,947
committed scales и только committed master weights. Независимый worst снизился
до `1.000346068`, normalized headroom вырос до `0.836528`. Следующий
контролируемый атом — `up_proj +0.78125%` из восстановленного frontier.
Этот атом прошёл оба holdout, добавил 98,304 hard-ternary weights и поднял
`up_proj` до `70.01953125%`. Development предпочёл linked BF16 MLP arm, но
независимый holdout выбрал candidate-only (`1.000616263` против
`1.000722317`). Normalized headroom равен `0.709288`; следующий round-robin
атом — `down_proj +0.390625%`.
Следующий down/gate/up цикл добавил ещё 172,032 hard-ternary weights. Coverage
последовательно достиг `6.93359375% down`, `3.3203125% gate` и
`70.80078125% up`; все шесть arm-а прошли два holdout, а каждая транзакция
независимо выбрала candidate-only. Финальный worst равен `1.000783641`,
normalized headroom — `0.631200`. Следующий round-robin атом —
`down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `4.98046875%`. Holdout выбрал candidate-only вместо
development-победителя linked arm (`1.000442839` против `1.000456372`),
normalized headroom равен `0.788461`. Следующий round-robin атом —
`gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.34375%`. Candidate-only выиграл у linked arm
(`1.000418275` против `1.000454195`), normalized headroom равен `0.800263`.
Следующий round-robin атом — `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `66.89453125%`. Candidate-only выиграл у linked arm
(`1.000487872` против `1.000599281`), normalized headroom равен `0.767662`.
Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `3.80859375%`. Linked arm выиграл у candidate-only по
независимому worst (`1.000555450` против `1.000598652`), normalized headroom
равен `0.731646`. Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.7578125%`. Candidate-only минимально выиграл у linked arm
по независимому worst (`1.000559869` против `1.000562804`), normalized
headroom равен `0.729604`. Следующий контролируемый атом —
`up_proj +1.5625%`; при отказе или tight headroom выполняется recovery.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `62.20703125%`. Candidate-only выиграл у linked arm по независимому worst
(`1.000713586` против `1.000765682`), normalized headroom равен `0.656314`.
Следующий round-robin атом — `down_proj +0.390625%`; при отказе или tight
headroom выполняется recovery.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `4.19921875%`. Linked BF16-MLP arm выиграл у candidate-only по
независимому worst (`1.000811028` против `1.000830712`), normalized headroom
равен `0.609651`. Следующий round-robin атом — `gate_proj +0.1953125%`, после
него выполняется masked proxy recovery перед следующим крупным up-атомом.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.953125%`. Linked BF16-MLP arm выиграл у candidate-only по
независимому worst (`1.000820492` против `1.000865155`), normalized headroom
равен `0.605231`. Следующий шаг — coverage-neutral masked proxy recovery,
после которого можно безопаснее пробовать следующий крупный up-атом.
Masked proxy recovery v5 сохранил coverage и committed masks, изменил только
два уже принятых ternary-кода и перенастроил shared scales. Независимый worst
снизился с `1.000820492` до `1.000171060`, normalized headroom вырос до
`0.917697`. Следующий контролируемый атом — `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `63.76953125%`. Linked BF16-MLP arm выиграл у candidate-only по
независимому worst (`1.000192249` против `1.000249673`), normalized headroom
равен `0.907755`. Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `4.58984375%`. Candidate-only минимально выиграл у linked arm
(`1.000247348` против `1.000264837`), normalized headroom равен `0.881399`.
Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.1484375%`. Candidate-only выиграл у linked arm
(`1.000247756` против `1.000354786`), normalized headroom равен `0.881245`.
Следующий round-robin атом — `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `65.33203125%`. Candidate-only минимально выиграл у linked arm
(`1.000363082` против `1.000371670`), normalized headroom равен `0.826441`.
Следующий round-robin атом — `down_proj +0.390625%`.
Этот gate-атом также прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.171875%`. Candidate-only выиграл и development, и holdout:
`1.000480310` против `1.000712729` у linked arm. При динамическом gate
`1.002041273` normalized headroom равен `0.764701`, поэтому следующий
round-robin атом — `up_proj +3.125%`.
Этот up-атом прошёл оба holdout, добавил 393,216 weights и поднял `up_proj`
до `57.51953125%`. Linked arm минимально выиграл у candidate-only по
независимому worst (`1.000862859` против `1.000871839`), normalized headroom
равен `0.579647`. Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `3.02734375%`. Linked arm выиграл (`1.000977218` против
`1.000998762`), normalized headroom равен `0.524267`. Следующий round-robin
атом — `gate_proj +0.1953125%`.

Последнее проверенное продолжение добавило девять транзакций после frontier
`s0022`: `down/gate s0017`, `up s0023`, полный цикл
`up s0024 -> down/gate s0018` и полный цикл
`up s0025 -> down/gate s0019`. Всего добавлено 516,096 hard-ternary weights.
Каждый accepted checkpoint был fresh-reload проверен на development suite и
на двух независимых holdout suites. Финальный `gate s0019` выбрал
`candidate_bf16_mlp` с worst `1.001126802` против `1.001156927` у
candidate-only; coverage обоих arm одинаков. Все три campaign-state
синхронизированы на checkpoint SHA
`49210f7ad73b22bb7a0beec96cf516ad33fcce6aab0e36cc3869bc2defef1930`.
Следующий round-robin атом — `up_proj +0.78125%` (`s0026`).

`s0026` был корректно отклонён на development gate и не изменил frontier.
Автоматически уменьшенный `up s0027 +0.390625%`, затем
`down s0020 +0.390625%` и `gate s0020 +0.1953125%` прошли оба holdout и
добавили 122,880 hard-ternary weights. Финальный checkpoint имеет SHA
`eeb870dbce2d65b73899945454bda00b22ee9e16a9808591aedb3f6d3e00fdf3`;
следующий атом — `up_proj s0028 +0.390625%`.

Продолжение `s0028..s0030`, `down s0021..s0023` и `gate s0021..s0022`
добавило 147,456 hard-ternary weights. Две слишком крупные транзакции
(`down s0021` и `up s0029`) были отклонены до изменения frontier; шесть
уменьшенных транзакций прошли fresh reload и оба holdout. Текущий checkpoint
имеет SHA `f61762fcefef7bac94b3cd9f2dbb1072087904500a109a2f66ef0145a2035215`,
coverage `74.12109375% up / 8.7890625% down / 4.39453125% gate` и worst
`1.001663819` при guide `1.002147694`. Normalized headroom `0.225300`, поэтому
следующий этап — coverage-neutral masked proxy recovery v7, затем минимальные
атомы `0.09765625%`.

Recovery v7 прошёл оба holdout без изменения coverage, committed masks и
непринятых BF16 weights. Он снизил worst до `1.001092666`. После него четыре
минимальные транзакции `up s0031`, `down s0024`, `gate s0023`, `up s0032`
добавили 49,152 hard-ternary weights и также прошли оба holdout. Текущий
checkpoint имеет SHA
`b5228f1ce45f292030042f1b2d504703b89404bf434b7c20780f73c1d55bb20f`,
coverage `74.31640625% up / 8.88671875% down / 4.4921875% gate`, worst
`1.001076137` при guide `1.002149122` и normalized headroom `0.499267`.
Следующий round-robin атом — `down_proj s0025 +0.09765625%`.

Следующий цикл `down s0025`, `gate s0024`, `up s0033` добавил ещё 36,864
hard-ternary weights и полностью прошёл два holdout. Все три раза выбран
candidate-only checkpoint. Текущий SHA —
`491fdf27f6b149637358d9b94454deb2ac03e92c6322a05165e1b112f57e4ff8`,
coverage `74.4140625% up / 8.984375% down / 4.58984375% gate`, worst
`1.001147515`, guide `1.002150193`, normalized headroom `0.466320`.
Следующий атом — `down_proj s0026 +0.09765625%`.

Следующий цикл `down s0026`, `gate s0025`, `up s0034` добавил ещё 36,864
hard-ternary weights. Все три кандидата прошли development, fresh reload и
два независимых holdout; выбран candidate-only. Новый SHA —
`2b49af314e17800f79743216b2365378daa69f9db0e91c76db605086617581f8`,
coverage `74.51171875% up / 9.08203125% down / 4.6875% gate`, worst
`1.001148209`, guide `1.002151265`, normalized headroom `0.466263`.
Следующий атом — `down_proj s0027 +0.09765625%`.

Цикл 27 (`down s0027`, `gate s0026`, `up s0035`) выполнен единым
round-robin orchestrator в одной терминальной сессии. Максимально одновременно
работала одна кампания; каждый accepted frontier синхронизирован во всех трёх
campaign state до удаления предыдущего checkpoint. Цикл добавил 36,864
hard-ternary weights и прошёл шесть независимых audit-запусков. Новый SHA —
`5f6f1c570b308767eb9cfca8b6761749427ddb65ccf9fa7f12b405d784c4db92`,
coverage `74.609375% up / 9.1796875% down / 4.78515625% gate`, worst
`1.001155576`, guide `1.002152336`, normalized headroom `0.463106`.
Следующий атом — `down_proj s0028 +0.09765625%`.

Также исправлена персистентность safe-streak адаптивного transaction sizer.
До исправления каждый одношаговый дочерний процесс начинал streak с нуля,
поэтому `grow_after=2` не мог сработать между round-robin циклами. Теперь
счётчик сохраняется в campaign state; 51 unit-тест проходит.

Три последовательных круга 28–30 добавили ещё 122,880 hard-ternary weights:
девять из девяти транзакций приняты. Это первая реальная проверка сохранённого
safe streak: `gate` после двух roomy-проходов вырос с `1/1024` до `1/512`, а
увеличенный `gate s0029` прошёл development, fresh reload и оба holdout.
Финальный SHA —
`3bf1f3e255a5d1f8c35a05af64a6484fa1e0b5508cee13226c50fb5bff3c0c07`,
coverage `74.90234375% up / 9.47265625% down / 5.17578125% gate`, worst
`1.001280233`, guide `1.002155907`. Следующие размеры: down/up `1/1024`,
gate `1/512`.

Круги 31–33 добавили ещё 147,456 hard-ternary weights; все девять транзакций
приняты. `gate` трижды использовал выросший атом `1/512`, после чего tight
headroom вернул его к `1/1024`. Финальный SHA —
`bea838d433c1d61ec51e8d40664250fefd32b6a2261e4a33478bb7f3f8286435`,
coverage `75.1953125% up / 9.765625% down / 5.76171875% gate`, worst
`1.001487571`, guide `1.002160193`. Следующие шаги: down `s0034`, gate
`s0033`, up `s0042`, каждый с атомом `1/1024`.

Новый paired moving-block bootstrap audit не изменил point-gate verdict, но
показал статистически неуверенный SQuAD margin. При блоках по 8 соседних окон
верхняя 95% граница ratio равна `1.006936` на audit-v3 и `1.003670` на
audit-v4, тогда как C4 и оба code-домена уверенно ниже guide. Поэтому перед
следующим ростом coverage выполняется hard-forward coverage-neutral recovery,
после чего те же frozen suites проверяются повторно. Старый frontier не
откатывается задним числом: confidence-policy не входила в его predeclared
acceptance gate.

После coverage-neutral recovery круги 34–36 приняли ещё девять транзакций.
Новый SHA —
`e46c6299a572218fdf8830dccdaffd47c3f84870a5268c3b0e65966d5423ceb1`,
coverage `75.48828125% up / 10.05859375% down / 6.0546875% gate`, общий
coverage модели `4.326813195%`. Точечные audit-v3/v4 проходят guide
`1.002163407`; paired CI уверенно пропускает C4/code, но всё ещё не SQuAD.
Основной checkpoint не откатывается, поскольку confidence gate не входил в
его predeclared policy.

Параллельно открыт безопасный `transform slot`: на нетронутой
`layer23.q_proj` fixed g128 RHT seed 307 уменьшил **инкрементальный**
worst-domain ущерб обычной тернаризации примерно на 90% на v3/v4. Эти ratios
считались относительно frontier `s0044`, поэтому сравнивать их напрямую с
кумулятивной BF16-guide было нельзя. Короткий hard-forward proxy/scale recovery
затем был выбран только на development и проверен кумулятивно: audit-v3 дал
SQuAD ratio `1.002892` и upper-95 ratio `1.008465`, audit-v4 — point
`0.998729`, но upper-95 `1.004656`. Кандидат отклонён, coverage не изменён.
RHT остаётся сильной инициализацией для будущих WLS/tail ablation, но не
является принятым transform commit. Будущий `llama.cpp` путь всё равно должен
уметь исполнять transform рядом с packed codes, а не материализовать BF16.

Поскольку v3/v4 многократно влияли на выбор arm и размер транзакций, дальше они
называются recurring validation. Для независимого доказательства нужны новые
непересекающиеся sealed v5/v6, которые не участвуют в настройке.

После остановки point-only роста проведён D8--D10 tail stress. Activation-WLS
сильно помог первой D10-группе `down_proj`, но на второй непересекающейся
реплике сырая WLS-инициализация оказалась хуже absmean. Рецепт второй реплики
был зафиксирован заранее и не менялся: hard-forward recovery сохранил codes,
улучшил только scales и прошёл prospective dual gate на v3/v4. Принято 96
групп (`12,288` weights), `down_proj` вырос до `10.44921875%`, общий coverage
до `4.333955508%`. Этот результат поддерживает scale recovery, но опровергает
гипотезу об универсальном превосходстве WLS; следующие реплики должны заранее
фиксировать initializer и оставаться непересекающимися.

Остались три MLP-матрицы. Component ablation показал:

1. `up_proj` — наименее вредная отдельная матрица;
2. `down_proj` и `gate_proj` сильнее ухудшают SQuAD;
3. пары MLP дают нелинейно больший ущерб.

One-shot 100% `up_proj` после hard proxy + continuous recovery дошёл на
audit-v3 до `0.998760 / 1.004714 / 0.978121`, но не прошёл заданный gate.
Incremental WAL нашёл безопасный путь: после 12.5% последовательные шаги
3.125%, 1.5625% и 0.78125% довели `up_proj` до 28.125% и прошли два audit
suite, тогда как крупные очередные шаги 12.5% и 6.25% откатились. Следующий
контроллер должен выбирать размер атома автоматически из holdout margin и
уменьшать его при rollback.

Добавлен новый `candidate_bf16_mlp` arm: candidate остаётся hard ternary, а
связанные ещё не конвертированные `down/gate` получают локальные непрерывные
BF16 окна с точным snapshot/rollback. В matched ablation он улучшил audit-v3
SQuAD с `1.001575` до `1.001471`, не увеличив ternary coverage `down/gate`.
Один linked шаг `+1.5625%` откатился при `1.002169`, но два последовательных
шага по `+0.78125%` достигли того же coverage и прошли оба holdout. Это
подтверждает path-dependence уже для новой компенсационной схемы.

После этого два linked-атома по `+0.390625%` и один адаптивно увеличенный атом
`+0.78125%` также прошли development, fresh reload и оба holdout-аудита. При
сужении audit-margin следующий атом снова уменьшается до `+0.390625%`.

Реальный adaptive campaign runner теперь выполняет recovery, fresh reload,
параллельные audit-v3/v4 для каждого прошедшего arm-а, holdout-selection,
coverage-aware resizing и crash-safe cleanup. Первый автоматический gate-шаг
поднял `gate_proj` до `0.1953125%`; linked BF16-down arm выиграл holdout у
candidate-only (`1.001475638` против `1.001515298`).

Для `down_proj` добавлен отдельный column-structured selector: один столбец
g128-групп соответствует одному входному 128-канальному SwiGLU-блоку и
однозначно задаёт локальные строки компенсации в `up/gate`. Первый атом из 96
групп (`12,288` weights, `0.09765625%` матрицы) прошёл оба holdout. В этом
случае candidate-only оказался немного лучше BF16-компенсации (`1.001527907`
против `1.001550445`) и стал новым frontier. Следующий атом был увеличен в
четыре раза до 384 групп и также прошёл оба holdout; на нём BF16-компенсация
уже выиграла (`1.001487769` против `1.001539105`). Контроллер сузил следующий
размер с `0.390625%` до `0.1953125%` из-за небольшого audit-margin. Этот
следующий атом также прошёл оба holdout, снова с небольшим преимуществом
BF16-компенсации (`1.001531213` против `1.001537834`), после чего контроллер
вернул следующий размер к минимуму `0.09765625%`.

Критерий перехода: принять все три MLP, получить второй полный block и
повторить неизменяемый cumulative audit.

## Фаза 5 — пройти decoder

Blocks выбираются по измеренной чувствительности, не по номеру. После каждого
commit sensitivity пересчитывается, потому что frontier изменился. Всегда
публикуются оба числа:

```text
структурно converted blocks / 28
independently accepted blocks / 28
```

Нельзя считать частичный tensor полноценным block и нельзя ослаблять audit
ради процента покрытия.

## Фаза 6 — tied embedding/LM head

Embedding/head содержит 311,164,928 weights и одновременно является входной
таблицей и output classifier. Для него нужен отдельный frequency-balanced
token suite, logit distillation и, возможно, mixed Q2/Q4 budget. Любое спасение
Q4 публикуется в true average bpw.

## Фаза 7 — full-model validation

- C4/Wiki/code и независимые языковые corpora;
- MMLU/ARC/HellaSwag/PIQA;
- GSM8K/MATH;
- HumanEval/MBPP;
- instruction following, JSON и tool calling;
- русский/казахский и long-context tests;
- несколько seeds и confidence intervals;
- BF16, INT4 и сильный Q2 PTQ baselines.

## Фаза 8 — packed export и llama.cpp

До фиксации representation менять `llama.cpp` преждевременно. После фиксации:

1. exporter пишет packed two-bit codes и FP16 g128 scales;
2. GGUF tensor type совпадает с layout;
3. loader не распаковывает полную BF16-матрицу;
4. CPU/CUDA/Metal dequant/GEMM paths проходят bit-exact tests;
5. измеряются real file size, resident memory и tok/s.

Если останется единый Q2-g128 layout, можно адаптировать существующий близкий
путь. Если внутри tensor будут Q2/Q4 blocks, residual plane или новый decoder,
потребуются новый tensor type, quantizer, loader и kernels.

## Фаза 9 — binary

От устойчивой ternary-модели нули постепенно переводятся в знаковые states.
Цель Q1-g128: `1 + 16/128 = 1.125 bpw`. Binary checkpoint и его quality target
всегда отделяются от ternary результата.
