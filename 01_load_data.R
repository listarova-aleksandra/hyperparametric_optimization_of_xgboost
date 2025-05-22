library(mlr3)             # ML-платформа (после успешной установки)
library(mlr3learners)     # Расширенные алгоритмы (XGBoost и др.)
library(data.table)       # Быстрое чтение и обработка таблиц

  # 2. Загрузка датасета из локального файла
  #    предполагается, что adult.data лежит в папке data/
  # =======================
local_file <- "data/adult.data"
if (!file.exists(local_file)) {
  stop(
    "Файл 'data/adult.data' не найден.\n",
    "Пожалуйста, скачай его по:\n",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data\n",
    "и положи в папку data/ твоего проекта."
  )
}

adult <- tryCatch(
  # читаем через read.csv и сразу в data.table
  as.data.table(
    read.csv(local_file,
             header = FALSE,
             na.strings = "?",
             strip.white = TRUE)
  ),
  error = function(e) stop("Ошибка чтения локального файла adult.data: ", e$message)
)

# Далее — присвоение имён столбцам и предобработка, как раньше:
if (ncol(adult) == 15) {
  setnames(adult, old = names(adult), new = c(
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
  ))
} else {
  stop("Ожидалось 15 колонок, но получено: ", ncol(adult))
}

# 6. Анализ наличия пропусков
na_counts <- sapply(adult, function(col) sum(is.na(col)))
print("Количество пропусков в каждом столбце:")
print(na_counts)

# 7. Удаление строк с пропусками (для простоты)
adult_clean <- na.omit(adult)
message("Удалено строк с пропусками: ", nrow(adult) - nrow(adult_clean))

# 8. Преобразование категориальных колонок и целевой переменной в факторы
cat_cols <- c(
  "workclass", "education", "marital_status", "occupation",
  "relationship", "race", "sex", "native_country", "income"
)
adult_clean[, (cat_cols) := lapply(.SD, as.factor), .SDcols = cat_cols]

# 9. Проверка структуры очищенного датасета
print(str(adult_clean))


