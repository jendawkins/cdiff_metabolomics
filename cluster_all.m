h = findall( 0, 'type', 'figure' );  % Find all figures
h.delete               

X = load('ALLDATA.mat');
targets = load('ALLLABELS_NW.mat');
met_names = load('names.mat');
met_targets = load('met_targets_ALL_big.mat');
met_targs = cellstr(met_targets.a);

targets_all = load('ALLLABELS_EVEN.mat');
data = X.a;

met_cats = [{'Amino Acid'}, {'Carbohydrate'},{'Cofactors and Vitamins'},...
    {'Energy'},{'Lipid'},{'Nucleotide'},{'Partially Characterized Molecules'},...
    {'Peptide'},{'Xenobiotics'}];

tstruct = struct();

colors0 = cell(length(targets_all.a),1);
colors0(:) = {'b'};

carray0 = cellstr(targets_all.a);
carray0 = cellstr(cellfun(@(x) x(1), carray0));
ixs = contains(carray0, 'R');

C    = cell(sum(ixs),1);
C(:) = {'y'};
colors0(ixs) = C;

colors1 = colors0;
carray = cellstr(targets.a);
carray2 = cellstr(cellfun(@(x) x(1), carray));
ixs = contains(carray, 'Recur');

C    = cell(sum(ixs),1);
C(:) = {'r'};

colors1(ixs) = C;

% C = cell(length(find(ixs==0)),1);
% C(:) = {'g'};
% colors2((ixs == 0)) = C;

carrayfin = cell(length(carray2),1);
for i = 1:length(carray2)
    if strcmp(carray2{i}, {'R'})
        carrayfin(i) = {'R'};
    elseif strcmp(carray0{i}, 'R') && strcmp(carray2{i}, {'C'})
        carrayfin(i) = {'E'};
    else 
        carrayfin(i) = {'C'};
    end
end

tstruct.Labels = carrayfin;
tstruct.Colors = colors1;


cmap_targs = colormap(jet(length(met_cats)));
colors3 = cell(length(met_targs),1);

cvec = cellstr(char('m','c','r','w','g','b','w','k','y'));
m_unique = met_cats;
for k = 1:size(cmap_targs,1)
    ixs = contains(met_targs, m_unique(k));
    C = cell(length(find(ixs==1)),1);
    C(:) = cvec(k);
    colors3(ixs==1) = C;
    disp([m_unique(k), cvec(k)])
end

tstruct2.Labels = met_targs;
tstruct2.Colors = colors3;

cg = clustergram(data.', 'RowPDist', 'spearman','ColumnPDist',...
    'spearman','Linkage','average','Cluster','all','LabelsWithMarkers','true',...
    'Standardize','row'); 
set(cg,'ColumnLabels',carrayfin,'RowLabels',met_targs)
% cg.LabelsWithMarkers = 'true';
cg.addXLabel('Patients')
cg.addYLabel('Metabolites')
cg.ColumnLabelsColor = tstruct;
cg.RowLabelsColor = tstruct2;
% cmap = get(cg_s,'Colormap');
set(gca,'fontsize', 20)



% X2 = rand(size(X))