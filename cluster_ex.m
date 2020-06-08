load filteredyeastdata
cgo = clustergram(yeastvalues(1:30,:),'Standardize','Row')
set(cgo,'RowLabels',genes(1:30),'ColumnLabels',times)