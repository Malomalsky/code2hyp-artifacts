# Interpretation Guide: AST Geometry Signal

Файл фиксирует интерпретацию результатов для статьи. Он не заменяет CSV-артефакты,
а задает корректный язык описания результата.

## Основной тезис

На выборке из 550 программ DTA локальные распределения дискретной кривизны AST
статистически связаны с типом вычислительной задачи. Эта связь сохраняется для
части curvature-признаков после линейного контроля размера AST и локального
роста окрестностей.

Строгая формулировка:

> Геометрический профиль AST содержит task-level структурный сигнал, который
> частично не сводится к размеру дерева и локальному ball-growth.

Нельзя писать:

> Кривизна сама по себе доказывает улучшение retrieval/classification.

Такой вывод требует отдельного downstream-эксперимента.

## Как читать рисунок

Основной рисунок:

```text
figures/article_geometry_interpretation_limit50.pdf
figures/article_geometry_interpretation_limit50.png
```

### Панель A

Панель A показывает стандартизованные средние значения geometry descriptors по
типам задач.

Интерпретация:

- строки — типы задач DTA;
- столбцы — признаки размера, локального роста и кривизны AST;
- цвет — z-score признака относительно других задач;
- красный цвет означает, что задача имеет значение признака выше среднего;
- синий цвет означает значение ниже среднего.

Смысл панели A: разные типы задач имеют разные геометрические профили AST. Это
визуальное подтверждение того, что geometry descriptors не являются полностью
однородными по датасету.

### Панель B

Панель B сравнивает raw task-effect и residual task-effect.

Использован показатель:

```text
eta_squared_task = SS_between(task_id) / SS_total
```

Для residual-версии сначала удаляется линейный вклад размера и локального роста:

```text
curvature_metric = beta_0 + beta_1 * node_count + beta_2 * ball_size_mean_r3 + epsilon
eta_squared_task_residual = SS_between(task_id, epsilon) / SS_total(epsilon)
```

Интерпретация:

- серый столбец показывает исходную связь признака с типом задачи;
- зеленый столбец показывает, что остается после контроля размера AST и
  `ball_size_mean_r3`;
- если зеленый столбец сильно падает, признак в основном размерозависим;
- если зеленый столбец сохраняется, признак несет дополнительный task-level
  сигнал.

Ключевой вывод панели B: `forman_mean` существенно проседает после контроля,
а curvature mass-фракции сохраняют заметный residual task-effect.

### Панель C

Панель C показывает две оси интерпретации:

```text
x = covariate R^2
y = residual eta_squared_task
```

Интерпретация:

- правый нижний сектор — признак в основном объясняется размером и ростом AST;
- верхняя область — признак сохраняет task-level сигнал после контроля;
- `forman_mean` попадает в size-driven область;
- `forman_negative_mass`, `ollivier_negative_mass` и
  `ollivier_near_zero_mass` остаются в области task-specific curvature.

Смысл панели C: средняя кривизна и распределение режимов кривизны ведут себя
по-разному. Для статьи более интересны именно массовые фракции локальной
кривизны, а не только среднее значение.

## Статистическая интерпретация

Нулевая модель:

```text
H0: соответствие между geometry descriptor и task_id случайно.
```

Проверка:

```text
случайная перестановка task_id между 550 программами
5000 перестановок
Holm-correction по raw и residual тестам
```

Основной CSV:

```text
reports/task_geometry_permutation_tests_limit50.csv
```

Кратко:

```text
p_Holm <= 0.0028
```

Формулировать нужно именно как `p_Holm <= 0.0028`, а не как точное p-value,
потому что минимальный достижимый permutation p-value при 5000 перестановках
равен:

```text
1 / (5000 + 1) = 0.00019996
```

## Сильная интерпретация

Сильная, но корректная формулировка:

> Полученные результаты показывают, что локальные режимы дискретной кривизны
> AST статистически связаны с типом вычислительной задачи. После контроля
> размера дерева и локального роста окрестностей связь сохраняется для
> распределительных curvature-признаков, что указывает на наличие
> структурного сигнала, не сводимого к простой размерности AST.

## Граница утверждения

Что этот результат доказывает:

- curvature descriptors не являются произвольными декоративными признаками;
- в AST присутствует измеримый task-level geometry signal;
- mass-фракции кривизны информативнее одной средней кривизны;
- часть сигнала сохраняется после контроля размера и локального роста.

Что этот результат пока не доказывает:

- что curvature descriptors улучшают downstream-модель на независимом датасете;
- что AST-кривизна является причинным фактором сложности задачи;
- что результаты автоматически переносятся на CFG, DFG или CPG;
- что один конкретный тип кривизны всегда лучше другого.

## Черновик подписи к рисунку

**Figure X. Task-level AST geometry profiles and residual curvature signal.**
Panel A shows standardized task-level means of AST geometry descriptors on the
550-program DTA atlas. Panel B compares raw task-level eta-squared with
residual eta-squared after linear controls for AST size and local ball growth.
Panel C separates size-driven curvature descriptors from descriptors retaining
task-specific residual signal. Permutation testing with Holm correction over
raw and residual descriptors gives `p_Holm <= 0.0028`. The figure supports a
structural, not downstream-performance, interpretation: local curvature
fractions carry task-level information not fully explained by AST size.
