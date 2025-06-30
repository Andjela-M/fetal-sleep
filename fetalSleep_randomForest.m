% analyze longitudinal sleep data & determinants by means of random forest
% SLOOP - Andjela Markovic, November 2023

load('..\maturationWorkspace.mat');
resPath='..\RESULTS\FIGURES\';

ratioFetalSleep=[meanDaytimeSleep(:,1)./meanNighttimeSleep(:,1)];
ratioNewbornSleep=[meanDaytimeSleep(:,2)./meanNighttimeSleep(:,2)];
ratio3moSleep=[meanDaytimeSleep(:,3)./meanNighttimeSleep(:,3)];
ratio6moSleep=[meanDaytimeSleep(:,4)./meanNighttimeSleep(:,4)];

% prepare data table to predict sleep at 3 months
mo3DetTab=table(sex',birth',mo3Breastfeeding',mo3Age',ratioFetalSleep,meanSRI',wristTempPhase',ratioNewbornSleep,meanMilkMel',meanStoolMel');
mo3DetTab.Properties.VariableNames={'sex','birthMode','breastfeeding','age3mo','ratioFetalSleep','meanSRI','tempPhase','ratioNewbornSleep','meanMilkMel','meanStoolMel'};

mo3DetTab.sex = categorical(mo3DetTab.sex);
mo3DetTab.birthMode = categorical(mo3DetTab.birthMode);
mo3DetTab.breastfeeding = categorical(mo3DetTab.breastfeeding);
continuousInd=4:size(mo3DetTab,2);

% initialize for random forest
nTrees = 100; % number of trees

% 10-fold cross-validation
rng(123);
cv = cvpartition(size(ratio3moSleep, 1), 'KFold', 10);  
oobErrorAll = zeros(cv.NumTestSets,1);
featureImportance = zeros(size(mo3DetTab, 2), cv.NumTestSets);

for i = 1:cv.NumTestSets
    % training data for this fold
    trainX = mo3DetTab(cv.training(i), :);
    trainY = ratio3moSleep(cv.training(i), :);
  
    % test data for this fold
    testX = mo3DetTab(cv.test(i), :);
    testY = ratio3moSleep(cv.test(i), :);

    % standardize test and training data based on distribution of traning
    % data only to avoid data leakage
    mu = nanmean(trainX{:, continuousInd});
    sigma = nanstd(trainX{:, continuousInd});
    trainX{:, continuousInd} = (trainX{:, continuousInd} - mu) ./ sigma;
    testX{:, continuousInd} = (testX{:, continuousInd} - mu) ./ sigma;

    % train the model
    nTrees = 100;  
    model = TreeBagger(nTrees, trainX, trainY, 'Method', 'regression', 'OOBPrediction', 'On', 'OOBPredictorImportance','on','MinLeafSize', 5);
    
    % predict on test data
    predictedY = predict(model, testX);
    
    % calculate and store the error for this fold
    oobErrorAll(i) = sqrt(nanmean((predictedY - testY).^2));  % Root Mean Squared Error (RMSE) for regression

    % calculate feature importance for this fold
    featureImportance(:, i) = model.OOBPermutedPredictorDeltaError;

end

meanFeatureImportance = mean(featureImportance, 2);
meanError = nanmean(oobErrorAll);
stdError = nanstd(oobErrorAll);

% filter out features with less than or equal to 0 importance
positiveImportanceIndices = find(meanFeatureImportance > 0);
meanFeatureImportance = meanFeatureImportance(positiveImportanceIndices);
normalizedImportance = meanFeatureImportance / sum(meanFeatureImportance);
[sortedImportance, sortedIndices] = sort(normalizedImportance, 'descend');
cumulativeImportance = cumsum(sortedImportance);
threshold = 0.95;
numFeaturesToSelect = find(cumulativeImportance >= threshold, 1, 'first');
selectedFeaturesIndices = positiveImportanceIndices(sortedIndices(1:numFeaturesToSelect));
sortedSelectedImportance = sort(normalizedImportance(sortedIndices(1:numFeaturesToSelect)), 'descend');

% plot feature importance
b = bar(sortedSelectedImportance, 'k');
ylabel('Proportion of total importance');
ylim([0 1]);
dim = [.5 .6 .3 .3];
str = ['Mean error=' num2str(meanError,'%.3f') '; SD=' num2str(stdError,'%.3f')];
annotation('textbox',dim,'String',str,'FitBoxToText','on');
set(gca, 'XTickLabel',model.PredictorNames(selectedFeaturesIndices));
saveas(gca,[resPath 'FeatureImportance_Ratio3mo_7to7norm.jpg']);
close;

% recalculate model for only selected features
mo3SelDetTab=mo3DetTab(:,selectedFeaturesIndices);
model = TreeBagger(nTrees, mo3SelDetTab, ratio3moSleep, 'Method', 'regression', 'OOBPrediction', 'On', 'MinLeafSize', 5, 'OOBPredictorImportance', 'On');
nFeatures=length(selectedFeaturesIndices);

% plot partial dependence
for j = 1:nFeatures
    [pd, xi] = partialDependence(model, j);
    plot(xi,pd,'k','LineWidth',2);
    ylabel('3-month day/night sleep ratio');
    saveas(gca,[resPath 'PDP_Ratio3mo_7to7norm_' model.PredictorNames{j} '.jpg']);
    close;
end