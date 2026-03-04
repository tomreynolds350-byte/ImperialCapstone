# Round 02 Plot Guide (Internal)

This is an internal, novice-friendly guide to explain what the PCA and scatter plots show for each function.
It stays descriptive: ranges, variance structure, and which dimensions are most associated with y in the current data.

## How to read the plots (quick primer)

- Scatter x1 vs x2 (colored by y): shows coverage in 2D; color gradients indicate where y is higher or lower.
- Dim vs y scatter: one plot per dimension; a visible slope suggests that dimension may influence y.
- PCA (2D/3D): compresses inputs into principal components; the % variance tells you how much of the input spread is captured.
- Correlation heatmap: linear association between each x and y (positive/negative).
- Parallel coordinates: all dimensions on one chart; color highlights whether higher y aligns with certain ranges.

## Function 1

- Samples: 12, Dimensions: 2
- y range: -0.0036 to 0.0000 (mean -0.0003, std 0.0010)
- x ranges: x1 0.0020 to 0.8839, x2 0.0787 to 0.9964
- PCA variance (PC1/PC2): 54.8%, 45.2%
- Strongest linear association with y (by |corr|): x1 (corr -0.138)
- Plots to review: deliverables/round_02/plots/post/function_1/scatter_x1_x2.png, deliverables/round_02/plots/post/function_1/dim_vs_y.png, deliverables/round_02/plots/post/function_1/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 2

- Samples: 12, Dimensions: 2
- y range: -0.0868 to 0.6112 (mean 0.1862, std 0.2294)
- x ranges: x1 0.0024 to 0.8778, x2 0.0287 to 0.9954
- PCA variance (PC1/PC2): 66.1%, 33.9%
- Strongest linear association with y (by |corr|): x1 (corr 0.545)
- Plots to review: deliverables/round_02/plots/post/function_2/scatter_x1_x2.png, deliverables/round_02/plots/post/function_2/dim_vs_y.png, deliverables/round_02/plots/post/function_2/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 3

- Samples: 17, Dimensions: 3
- y range: -0.3989 to -0.0348 (mean -0.1240, std 0.1048)
- x ranges: x1 0.0468 to 0.9660, x2 0.0574 to 0.9679, x3 0.0661 to 0.9909
- PCA variance (PC1/PC2/PC3): 41.7%, 33.9%, 24.4%
- Strongest linear association with y (by |corr|): x3 (corr -0.636)
- Plots to review: deliverables/round_02/plots/post/function_3/scatter_x1_x2.png, deliverables/round_02/plots/post/function_3/dim_vs_y.png, deliverables/round_02/plots/post/function_3/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 4

- Samples: 32, Dimensions: 4
- y range: -32.6257 to -0.1254 (mean -17.1511, std 7.8563)
- x ranges: x1 0.0378 to 0.9856, x2 0.0063 to 0.9196, x3 0.0422 to 0.9626, x4 0.0815 to 0.9995
- PCA variance (PC1/PC2/PC3): 35.4%, 25.6%, 22.5%
- Strongest linear association with y (by |corr|): x4 (corr -0.549)
- Plots to review: deliverables/round_02/plots/post/function_4/scatter_x1_x2.png, deliverables/round_02/plots/post/function_4/dim_vs_y.png, deliverables/round_02/plots/post/function_4/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 5

- Samples: 22, Dimensions: 4
- y range: 0.1129 to 4665.9657 (mean 552.5428, std 1290.7094)
- x ranges: x1 0.1199 to 0.9545, x2 0.0382 to 0.9300, x3 0.0889 to 0.9527, x4 0.0729 to 0.9576
- PCA variance (PC1/PC2/PC3): 42.1%, 22.6%, 20.3%
- Strongest linear association with y (by |corr|): x3 (corr 0.552)
- Plots to review: deliverables/round_02/plots/post/function_5/scatter_x1_x2.png, deliverables/round_02/plots/post/function_5/dim_vs_y.png, deliverables/round_02/plots/post/function_5/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 6

- Samples: 22, Dimensions: 5
- y range: -2.5712 to -0.6185 (mean -1.4731, std 0.4738)
- x ranges: x1 0.0028 to 0.9577, x2 0.1144 to 0.9319, x3 0.0165 to 0.9788, x4 0.0423 to 0.9617, x5 0.0049 to 0.8928
- PCA variance (PC1/PC2/PC3): 32.3%, 26.6%, 16.4%
- Strongest linear association with y (by |corr|): x4 (corr 0.588)
- Plots to review: deliverables/round_02/plots/post/function_6/scatter_x1_x2.png, deliverables/round_02/plots/post/function_6/dim_vs_y.png, deliverables/round_02/plots/post/function_6/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 7

- Samples: 32, Dimensions: 6
- y range: 0.0027 to 1.3650 (mean 0.2482, std 0.3540)
- x ranges: x1 0.0411 to 0.9425, x2 0.0118 to 0.9620, x3 0.0036 to 0.9246, x4 0.0002 to 0.9610, x5 0.0149 to 0.9987, x6 0.0511 to 0.9510
- PCA variance (PC1/PC2/PC3): 26.4%, 24.0%, 18.2%
- Strongest linear association with y (by |corr|): x1 (corr -0.369)
- Plots to review: deliverables/round_02/plots/post/function_7/scatter_x1_x2.png, deliverables/round_02/plots/post/function_7/dim_vs_y.png, deliverables/round_02/plots/post/function_7/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 8

- Samples: 42, Dimensions: 8
- y range: 5.5922 to 9.7766 (mean 7.8303, std 0.9937)
- x ranges: x1 0.0091 to 0.9859, x2 0.0034 to 0.9740, x3 0.0229 to 0.9989, x4 0.0090 to 0.9030, x5 0.0096 to 0.9869, x6 0.0221 to 0.9902, x7 0.0276 to 0.9929, x8 0.0339 to 0.9888
- PCA variance (PC1/PC2/PC3): 20.0%, 18.6%, 15.3%
- Strongest linear association with y (by |corr|): x3 (corr -0.662)
- Plots to review: deliverables/round_02/plots/post/function_8/scatter_x1_x2.png, deliverables/round_02/plots/post/function_8/dim_vs_y.png, deliverables/round_02/plots/post/function_8/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.
