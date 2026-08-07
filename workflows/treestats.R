library(ape)
library(treestats)
library(treespace)
library(optparse)
library(tidyverse)
library(adegraphics)

outdir <- parse_args(OptionParser(option_list = list(make_option(c("--path", type="character")))))$path
tree_files <- list.files(path = outdir, pattern = "sim.trees$", recursive = TRUE, full.names = TRUE)

# read trees from remaster log
read_remaster <- function(filepath, min_tips = 3) {
  txt <- paste(readLines(filepath), collapse = "\n")
  tree_lines <- grep("^\\s*tree\\s+", strsplit(gsub("\\[[^]]*\\]", "", gsub("\r\n|\r", "\n", txt)), "\n")[[1]], ignore.case = TRUE, value = TRUE)
  newick <- sub("^[^=]*=\\s*", "", tree_lines)
  keep <- lengths(regmatches(newick, gregexpr(",", newick, fixed = TRUE))) + 1 >= min_tips
  trees <- lapply(newick[keep], function(t) ape::read.tree(text = t))
  names(trees) <- seq_along(trees)
  class(trees) <- "multiPhylo"
  return(trees)
}

# calculate summary statistics
stats_list <- list()
all_trees <- list()
for (file in tree_files) {
  trees <- read_remaster(file)
  for (tree in trees) {
    stats_list[[length(stats_list) + 1]] <- data.frame(
      group = basename(dirname(file)),
      colless = treestats::colless(tree), 
      cherries = treestats::cherries(tree),
      average_leaf_depth = treestats::average_leaf_depth(tree), 
      mean_pair_dist = treestats::mean_pair_dist(tree),
      tree_height = treestats::tree_height(tree), 
      mean_branch_length = treestats::mean_branch_length(tree),
      var_branch_length = treestats::var_branch_length(tree)
    )
  }
  all_trees <- c(all_trees, trees)
}
class(all_trees) <- "multiPhylo"
stats_df <- do.call(rbind, stats_list)

# boxplots
df_long <- stats_df %>% pivot_longer(cols = -group, names_to = "param", values_to = "val")
ggplot(df_long, aes(x = group, y = val, fill = group)) +
  geom_boxplot(outlier.alpha = 0.5) +
  facet_wrap(~ param, scales = "free_y", nrow = 2, ncol = 4) +
  theme_minimal() +
  theme(legend.position = c(0.9, 0.25), axis.text.x = element_blank(), axis.title.x = element_blank(), axis.title.y = element_blank())
ggsave(file.path(outdir, "boxplots.png"), width = 8, height = 4)

# treespace
space = treespace::treespace(all_trees, nf=3, lambda=0.5)
g <- plotGroves(
  space$pco, groups = stats_df$group, type = "ellipse", 
  starSize = 0, point.cex = 0.5, plabels.cex = 0, plegend.size = 1, 
  col.pal = function(n) {hcl(h = seq(15, 375, length = n + 1), l = 65, c = 100)[1:n]}
)
pdf(file.path(outdir, "treespace.pdf"))
print(g[[1]])
dev.off()
