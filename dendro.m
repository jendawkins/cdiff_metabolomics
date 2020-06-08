h = findall( 0, 'type', 'figure' );  % Find all figures
h.delete               

X = load('week_one.mat');
targets = load('targets_dict.mat');
met_names = load('names.mat');
met_targets = load('met_targets_w1.mat');
met_targs = cellstr(met_targets.a);

targets1 = targets.a1_0;

met_cats = [{'Amino Acid'}, {'Carbohydrate'},{'Cofactors and Vitamins'},...
    {'Energy'},{'Lipid'},{'Nucleotide'},{'Partially Characterized Molecules'},...
    {'Peptide'},{'Xenobiotics'}];

data = X.a;

tstruct = struct();
tstruct.Labels = targets.week_one;

colors1 = cell(length(targets.week_one),1);

carray = cellstr(targets.week_one);
carray2 = cellstr(cellfun(@(x) x(1), carray));
ixs = contains(carray, 'Recur');

C    = cell(sum(ixs),1);
C(:) = {'y'};

colors1(ixs) = C;
colors2 = colors1;

C = cell(length(find(ixs==0)),1);
C(:) = {'b'};
colors2((ixs == 0)) = C;
carray2 = cellstr(cellfun(@(x) strrep(x,'R','E'), carray2));

colors_f = colors2;
carray3 = cellstr(cellfun(@(x) x(1), cellstr(targets1)));
ixs_r = contains(carray3, 'R');

C    = cell(sum(ixs_r),1);
C(:) = {'r'};

colors_f((ixs_r==1)) = C;
carrayfin = cell(length(carray3),1);
for i = 1:length(carray3)
    if strcmp(carray2{i}, {'E'}) && strcmp(carray3{i}, {'R'})
        carrayfin(i) = {'R'};
    elseif strcmp(carray2{i}, 'E') && strcmp(carray3{i}, {'C'})
        carrayfin(i) = {'E'};
    else 
        carrayfin(i) = {'C'};
    end
end

tstruct.Labels = carrayfin;
tstruct.Colors = colors_f;


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

cg = clustergram(data.', 'RowPDist', 'correlation','ColumnPDist',...
    'correlation','Linkage','average','OptimalLeafOrder','true','Cluster',...
    'all','LabelsWithMarkers','true','Standardize','col'); 
set(cg,'ColumnLabels',carrayfin,'RowLabels',met_targs)
% cg.LabelsWithMarkers = 'true';
cg.addXLabel('Patients')
cg.addYLabel('Metabolites')
cg.ColumnLabelsColor = tstruct;
cg.RowLabelsColor = tstruct2;
% cmap = get(cg_s,'Colormap');
set(gca,'fontsize', 30)

% data_n = normalize(data.', 1);
% h = heatmap(data_n,'Colormap',cmap);
% 
% caxis(h, [-3,3])

% X2 = rand(size(X))