library(ape)
library(treestats)
library(treespace)
library(optparse)
library(tidyverse)
library(adegraphics)

outdir <- parse_args(OptionParser(option_list = list(make_option(c("--path", type="character")))))$path
tree_files <- list.files(path = outdir, pattern = "sim_fixed_size.trees$", recursive = TRUE, full.names = TRUE)

# read trees from remaster log
read_remaster <- function(filepath, min_tips = 3) {
  txt <- paste(readLines(filepath), collapse = "\n")
  tree_lines <- grep("^\\s*tree\\s+", strsplit(gsub("\\[[^]]*\\]", "", gsub("\r\n|\r", "\n", txt)), "\n")[[1]], ignore.case = TRUE, value = TRUE)
  newick <- sub("^[^=]*=\\s*", "", tree_lines)
  trees <- lapply(newick, function(t) ape::read.tree(text = t))
  names(trees) <- seq_along(trees)
  class(trees) <- "multiPhylo"
  print(trees)
  return(trees)
}

# calculate summary statistics
stats_list <- list()
all_trees <- list()
psi_list <- list()
for (file in tree_files) {
  trees <- read_remaster(file)
  group <- basename(dirname(file))
  parts <- strsplit(group, "[psi]")[[1]]
  psi <- parts[5]
  for (tree in trees) {
    stats_list[[length(stats_list) + 1]] <- data.frame(
      group = paste0("psi = ", psi, ", p = ", gsub('.{1}$', '', parts[2])),
      colless = treestats::colless(tree, normalization = "yule"), 
      cherries = treestats::cherries(tree, normalization = "yule"), 
      average_leaf_depth = treestats::average_leaf_depth(tree, normalization = "yule"), 
      mean_pair_dist = treestats::mean_pair_dist(tree, normalization = "tips"),
      tree_height = treestats::tree_height(tree), 
      mean_branch_length = treestats::mean_branch_length(tree),
      var_branch_length = treestats::var_branch_length(tree)
    )
  }
  all_trees[[as.character(psi)]] <- c(all_trees[[as.character(psi)]], trees)
  psi_list <- unique(c(psi_list, as.character(psi)))
}
stats_df <- do.call(rbind, stats_list)
for (psi in psi_list){
  class(all_trees[[psi]]) <- "multiPhylo"
}

# boxplots
df_long <- stats_df %>% pivot_longer(cols = -group, names_to = "param", values_to = "val")
ggplot(df_long, aes(x = group, y = val, fill = group)) +
  geom_boxplot(outlier.alpha = 0.5) +
  facet_wrap(~ param, scales = "free_y", nrow = 2, ncol = 4) +
  labs(title = "Transmission tree summary statistics (fixed sample size)") +
  theme_minimal() +
  theme(legend.position = c(0.9, 0.25), axis.text.x = element_blank(), axis.title.x = element_blank(), axis.title.y = element_blank())
ggsave(file.path(outdir, "boxplots_fixed_size.png"), width = 8, height = 4)

# treespace
stats_df$psi <- stats_df
for (psi in psi_list) {
  space <- treespace::treespace(all_trees[[psi]], nf = 3, lambda = 0.5)
  pco <- as.data.frame(space$pco$li)
  names(pco)[1:2] <- c("PCo1", "PCo2")
  df_p <- stats_df[sub(",.*", "", stats_df$group) == paste0("psi = ", psi), ]
  pco$group <- df_p$group
  eig <- space$pco$eig
  perc_var_explained <- 100 * eig / sum(eig)

  ggplot(pco, aes(PCo1, PCo2, colour = group, fill = group)) +
    stat_ellipse(geom = "polygon", alpha = 0.15, level = 0.95, linewidth = 0.5) +
    geom_point(size = 0.7, alpha = 0.5) +
    labs(x = paste0("PCo1 (", round(perc_var_explained[1], 1), "%)"), y = paste0("PCo1 (", round(perc_var_explained[2], 1), "%)")) +
    theme_minimal() +
    theme(legend.position = "right", legend.title = element_blank(), panel.grid.minor = element_blank())
  ggsave(file.path(outdir, paste0("treespace_fixed_size_", psi, ".pdf")), width = 6, height = 5)
}
