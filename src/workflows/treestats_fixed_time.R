library(ape)
library(treeio)
library(treestats)
library(treespace)
library(optparse)
library(tidyverse)
library(adegraphics)

outdir <- parse_args(OptionParser(option_list = list(make_option(c("--path", type="character")))))$path
tree_files <- list.files(path = outdir, pattern = "sim_fixed_time.trees$", recursive = TRUE, full.names = TRUE)

# read trees from remaster log
read_remaster <- function(file) {
  trees <- lapply(treeio::read.beast(file), ape::as.phylo)
  class(trees) <- "multiPhylo"
  group <- basename(dirname(file))
  parts <- strsplit(group, "[psi]")[[1]]
  psi <- parts[5]
  file_stats_list <- lapply(trees, function(tree) {
    data.frame(
      group = paste0("psi = ", psi, ", p = ", gsub('.{1}$', '', parts[2])),
      colless = treestats::colless(tree, normalization = "yule"), 
      cherries = treestats::cherries(tree, normalization = "yule"), 
      average_leaf_depth = treestats::average_leaf_depth(tree, normalization = "yule"), 
      mean_pair_dist = treestats::mean_pair_dist(tree, normalization = "tips"),
      tree_height = treestats::tree_height(tree), 
      mean_branch_length = treestats::mean_branch_length(tree),
      var_branch_length = treestats::var_branch_length(tree)
    )
  })
  file_stats_df <- do.call(rbind, file_stats_list)
  return(list(stats = file_stats_df, trees = trees))
}
results <- lapply(tree_files, read_remaster)
stats_df <- do.call(rbind, lapply(results, `[[`, "stats"))
all_trees <- do.call(c, lapply(results, `[[`, "trees"))
class(all_trees) <- "multiPhylo"

# boxplots
df_long <- stats_df %>% pivot_longer(cols = -group, names_to = "param", values_to = "val")
ggplot(df_long, aes(x = group, y = val, fill = group)) +
  geom_boxplot(outlier.alpha = 0.5) +
  facet_wrap(~ param, scales = "free_y", nrow = 2, ncol = 4) +
  labs(title = "Transmission tree summary statistics (fixed sampling time)") +
  theme_minimal() +
  theme(legend.position = c(0.9, 0.25), axis.text.x = element_blank(), axis.title.x = element_blank(), axis.title.y = element_blank())
ggsave(file.path(outdir, "boxplots_fixed_time.png"), width = 8, height = 4)
