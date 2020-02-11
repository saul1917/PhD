function statsAnalysis()
    close all;
    csvPath{1} = 'Results100/inception_v3_20191129-223732.csv';
    csvPath{2} = 'Results100/resnet50_20191129-065906.csv';
    csvPath{3} = 'Results100/resnet18_20191128-231426.csv';
    for i = 1:3
        [descriptiveStats{i}.meanAccuracy, descriptiveStats{i}.stdAccuracy, descriptiveStats{i}.hAccuracy, descriptiveStats{i}.meanSensitivity, descriptiveStats{i}.stdSensitivity, descriptiveStats{i}.hSensitivity, descriptiveStats{i}.meanSpecificity, descriptiveStats{i}.stdSpecificity, descriptiveStats{i}.hSpecificity] = testKS_normality(csvPath{i});
    end
    %disp(csvPath{1})
    %disp(csvPath{2})
    %disp(csvPath{3})
    %testANOVA_acc(csvPath);
    testANOVA_sens(csvPath);
    %testANOVA_sens(csvPath)
    %testANOVA_acc(csvPath);
end

function [meanAccuracy, stdAccuracy, hAccuracy, meanSensitivity, stdSensitivity, hSensitivity, meanSpecificity, stdSpecificity, hSpecificity] = testKS_normality(csvPath)
    T = readtable(csvPath);
    accuracies = T.Accuracy;
    meanAccuracy = mean(accuracies);
    stdAccuracy = std(accuracies);
    standarizedAccuracies = (accuracies - meanAccuracy) / stdAccuracy;
    hAccuracy = kstest(standarizedAccuracies);
    %Sensitivity
    sensitivities = T.Sensitivity;
    meanSensitivity = mean(sensitivities);
    stdSensitivity = std(sensitivities);
    standarizedSensitivity = (sensitivities - meanSensitivity)/stdSensitivity;
    hSensitivity = kstest(standarizedSensitivity);
    %Specificity
    specificities = T.Specificity;
    meanSpecificity = mean(specificities);
    stdSpecificity = std(specificities);
    standarizedSpecificities = (specificities - meanSpecificity)/stdSpecificity;
    hSpecificity = kstest(standarizedSpecificities);
    %cdfplot(accuracies)
end

function testANOVA_acc(csvPaths)
    
    T1 = readtable(csvPaths{1});
    accuracies1 = T1.Accuracy;
    T2 = readtable(csvPaths{2});
    accuracies2 = T2.Accuracy;
    T3 = readtable(csvPaths{3});
    accuracies3 = T3.Accuracy;
    Accuracies = [accuracies1  accuracies2 accuracies3];
    p = anova1(Accuracies)
end

function testANOVA_sens(csvPaths)
    T1 = readtable(csvPaths{1});
    accuracies1 = T1.Sensitivity;
    T2 = readtable(csvPaths{2});
    accuracies2 = T2.Sensitivity;
    T3 = readtable(csvPaths{3});
    accuracies3 = T3.Sensitivity;
    Accuracies = [accuracies1  accuracies2 accuracies3];
    p = anova1(Accuracies)
end

function testANOVA_spec(csvPaths)
    T1 = readtable(csvPaths{1});
    accuracies1 = T1.Specificity;
    T2 = readtable(csvPaths{2});
    accuracies2 = T2.Specificity;
    T3 = readtable(csvPaths{3});
    accuracies3 = T3.Specificity;
    
    Specificities = [accuracies1  accuracies2 accuracies3];
    p = anova1(Specificities)
end
