# Создание задачи классификации и определение Learner + ParamSet для гиперпараметрической оптимизации

# ================================================
# 0. Установка необходимых пакетов (однократно)
#install.packages(c("mlr3", "mlr3learners", "paradox", "data.table", "xgboost"), dependencies = TRUE)
# ================================================

# 1. Подключение библиотек
library(mlr3)             # Основная платформа
library(mlr3learners)     # Дополнительные алгоритмы (XGBoost, Ranger)
library(data.table)       # Быстрое чтение таблиц
library(paradox)          # Описание пространства гиперпараметров

# 2. Загрузка подготовленных данных
if (file.exists("data/adult_clean.csv")) {
  adult_clean <- fread("data/adult_clean.csv")
} else if (exists("adult_clean")) {
  message("Используем объект adult_clean из глобальной среды")
} else {
  stop("Данные adult_clean не найдены. Сначала запусти scripts/01_load_data.R и сохрани результат в data/adult_clean.csv")
}

# 3. Создание задачи классификации
# Бинарная классификация: цель — фактор income c уровнями "<=50K" и ">50K"
task <- TaskClassif$new(
  id = "adult_income",
  backend = adult_clean,
  target = "income",
  positive = ">50K"
)

# 4. Вывод информации о задаче
cat("--- TaskClassif ---\n")
print(task)

# 5. Определение Learner
# Проверяем наличие XGBoost
if (!"classif.xgboost" %in% mlr_learners$keys()) {
  stop("Learner 'classif.xgboost' не найден. Установи пакет xgboost и перезапусти.")
}
learner <- lrn("classif.xgboost",
               objective = "binary:logistic",
               eval_metric = "auc",
               nthread = 2)

cat("\n--- Learner: classif.xgboost ---\n")
print(learner)

# 6. Описание пространства гиперпараметров (paradox v0.12+ style)
ps <- ps(
  eta = p_dbl(lower = 0.01, upper = 0.3),           # скорость обучения
  max_depth = p_int(lower = 3, upper = 10),          # максимальная глубина дерева
  nrounds = p_int(lower = 50, upper = 500),          # число итераций бустинга
  colsample_bytree = p_dbl(lower = 0.5, upper = 1.0),# доля признаков для сплита
  subsample = p_dbl(lower = 0.5, upper = 1.0)        # доля объектов для дерева
)

cat("\n--- ParamSet ---\n")
print(ps)

# 7. Сохранение объектов для следующих шагов
save(task, learner, ps, file = "data/task_learner_paramset.RData")
message("Задача, Learner и ParamSet сохранены в data/task_learner_paramset.RData")

