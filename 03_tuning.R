# Оптимизированный скрипт настройки гиперпараметров с параллелизмом

# ================================================
# 0. Установка зависимостей (однократно)
# install.packages(c("mlr3", "mlr3learners", "mlr3tuning", "mlr3mbo", "mlr3misc", "mlr3pipelines", "DiceKriging", "future", "data.table"), dependencies = TRUE)
# ================================================

# 1. Подключение библиотек и включение параллелизма
library(future)               # для параллельных вычислений
plan(multisession, workers = parallel::detectCores() - 1)  # используем все ядра, кроме одного

library(mlr3)
library(mlr3tuning)
library(mlr3mbo)
library(mlr3learners)
library(mlr3misc)
library(mlr3pipelines)    # кодирование категориальных
library(data.table)
library(DiceKriging)       # необходимость для mlr3mbo

library(future)               # для параллельных вычислений
plan(multisession, workers = parallel::detectCores() - 1)  # используется все ядра, кроме одного

library(mlr3)
library(mlr3tuning)
library(mlr3mbo)
library(mlr3learners)
library(mlr3misc)
library(mlr3pipelines)    # кодирование категориальных
library(data.table)

# 2. Загрузка task, learner, ps
load("data/task_learner_paramset.RData")  # загружает: task, learner, ps

# 3. Создание GraphLearner с one-hot для работы с факторами
graph_learner = po("encode", method = "one-hot") %>>% learner
graph_learner = GraphLearner$new(graph_learner)
graph_learner$predict_type = "prob"

# 4. Настройка ресемплинга и метрики
resampling = rsmp("cv", folds = 3)   # уменьшили до 3 fold для ускорения
measure = msr("classif.auc")

# 5. Определение пространства поиска
# Параметры XGBoost с префиксом classif.xgboost.
search_space = ps(
  classif.xgboost.eta           = p_dbl(lower = 0.01, upper = 0.3),
  classif.xgboost.max_depth    = p_int(lower = 3,    upper = 10),
  classif.xgboost.nrounds      = p_int(lower = 50,   upper = 300),  # уменьшили верхнюю границу
  classif.xgboost.colsample_bytree = p_dbl(lower = 0.5, upper = 1),
  classif.xgboost.subsample    = p_dbl(lower = 0.5,   upper = 1)
)

# 6. Функция для запуска настройки
run_tuning = function(tuner, name, n_evals = 30) {
  inst = TuningInstanceBatchSingleCrit$new(
    task         = task,
    learner      = graph_learner,
    resampling   = resampling,
    measure      = measure,
    search_space = search_space,
    terminator   = trm("evals", n_evals = n_evals)
  )
  message(sprintf("=== Запуск %s: до %d evals, %d-fold CV ===", name, n_evals, resampling$folds))
  tuner$optimize(inst)
  # Собираем результаты
  best_y = inst$result_y
  n_done = nrow(inst$archive$data)
  message(sprintf("%s: лучшая AUC = %.4f после %d evals", name, best_y, n_done))
  return(inst)
}

# 7. Запуск Grid Search
grid_tuner = tnr("grid_search", resolution = 4)
inst_grid   = run_tuning(grid_tuner, "Grid Search", n_evals = 64) #64

# 8. Запуск Random Search
rand_tuner  = tnr("random_search", batch_size = 1)
inst_rand   = run_tuning(rand_tuner, "Random Search", n_evals = 30) #30

# 9. Запуск Bayesian Optimization
mbo_tuner   = tnr("mbo")
inst_mbo    = run_tuning(mbo_tuner, "Bayesian Optimization", n_evals = 30) # 30

# 10. Сохранение результатов
save(inst_grid, inst_rand, inst_mbo, file = "data/tuning_results.RData")
message("Результаты настройки сохранены в data/tuning_results.RData")

