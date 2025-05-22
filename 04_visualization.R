# Визуализация и сравнение результатов настройки гиперпараметров

# 1. Подключение библиотек
library(data.table)  # для работы с таблицами
library(ggplot2)     # для построения графиков

# 2. Проверка и создание директорий для вывода
dir.create("plots", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)

# 3. Загрузка результатов настройки
results_file <- "data/tuning_results.RData"
if (!file.exists(results_file)) {
  stop("Файл tuning_results.RData не найден. Сначала запусти scripts/03_tuning.R и дождись его завершения.")
}
load(results_file)  # загружает: inst_grid, inst_rand, inst_mbo

# 4. Извлечение архивов в data.tables и добавление столбца eval (номер итерации)
arch_grid <- as.data.table(inst_grid$archive$data)
arch_rand <- as.data.table(inst_rand$archive$data)
arch_mbo  <- as.data.table(inst_mbo$archive$data)

arch_grid[, method := "Grid"]
arch_rand[, method := "Random"]
arch_mbo[, method := "Bayesian"]

# Добавляем номер оценки (eval) по порядку в каждой таблице
arch_grid[, eval := seq_len(.N)]
arch_rand[, eval := seq_len(.N)]
arch_mbo[, eval := seq_len(.N)]

# 5. Объединение данных в один data.table
arch_all <- rbindlist(list(arch_grid, arch_rand, arch_mbo), use.names = TRUE, fill = TRUE)

# 6. Построение графика сходимости AUC
p <- ggplot(arch_all, aes(x = eval, y = classif.auc, color = method)) +
  geom_line() +
  geom_point(size = 1) +
  labs(
    title = "Сходимость методов настройки гиперпараметров",
    x = "Номер оценки (eval)",
    y = "AUC на перекрёстной проверке",
    color = "Метод"
  ) +
  theme_minimal()

# 7. Сохранение графика
ggsave(
  filename = "plots/tuning_convergence.png",
  plot = p,
  width = 8,
  height = 5,
  units = "in"
)
message("График сохранён в plots/tuning_convergence.png")

# 8. Подготовка таблицы лучших результатов
best_results <- rbind(
  data.table(method = "Grid",    AUC = inst_grid$result_y, evals = nrow(inst_grid$archive$data)),
  data.table(method = "Random",  AUC = inst_rand$result_y, evals = nrow(inst_rand$archive$data)),
  data.table(method = "Bayesian",AUC = inst_mbo$result_y,  evals = nrow(inst_mbo$archive$data))
)

# 9. Сохранение таблицы и вывод
fwrite(best_results, file = "output/best_tuning_results.csv")
print(best_results)
message("Таблица лучших результатов сохранена в output/best_tuning_results.csv")
