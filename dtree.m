clear % clearing the wokspace
clc % clearing the terminal
close all % closing all open windows

% loading the dataset
dataset= readtable('diabetes.csv');
dataset.Outcome=grp2idx(dataset.Outcome);
X=dataset{:,1:8};
y=dataset{:,9};

% training
max_depth =4;
n_classes_ = size(unique(y),1);
n_features_ = size(X,2);
tree_ = grow_tree(X, y, max_depth,n_features_,n_classes_,0);

% testing
X_test=[6 148 72 35 0 33.6000 0.6270 50;6 148 72 35 0 33.6000 0.6270 50];
test=size(X_test,1);
predict=zeros(test,1);
for inputs=1:test
    node = tree_;
    while ~isempty(node.left)
        if X_test(inputs,node.feature_index) < node.threshold
            node = node.left;
        else
            node = node.right;
        end
    end
    predict(inputs)=node.predicted_class;
end
disp(predict);

% making the tree
function node= grow_tree(X, y, max_depth,n_features_,n_classes_, depth)
    num_samples_per_class=zeros(1,n_classes_);
    for i=1:n_classes_
        num_samples_per_class(i)=sum(y==i);
    end
    [~,predicted_class]= max(num_samples_per_class);
    node.predicted_class=predicted_class;
    node.left=[];
    if depth < max_depth
        best_idx=-1;best_thr = -1;
        m = size(y,1);
        if m > 1
            num_parent=zeros(1,n_classes_);
            for c=1:n_classes_
                num_parent(c)=sum(y==c);
            end
            s=0;
            for x=1:n_classes_
                 s=s+ (num_parent(x) / m)^2;
            end
            best_gini=1.0-s;
            for idx=1:n_features_
                temp=[y X(:,idx)];
                temp=sortrows(temp,2);
                classes=temp(:,1);
                thresholds=temp(:,2);
                num_left = zeros(1,n_classes_);
                num_right = num_parent;
                for i=2:m
                    c = classes(i - 1);
                    num_left(c) =num_left(c)+ 1;
                    num_right(c)=num_right(c)- 1;
                    sl=0;sr=0;
                    for x=1:n_classes_
                         sl=sl+ (num_left(x) / i)^2;
                         sr=sr+ (num_right(x) / (m - i))^2;
                    end
                    gini_left=1.0-sl;
                    gini_right=1.0-sr;
                    gini = (i * gini_left + (m - i) * gini_right) / m;
                    if thresholds(i) == thresholds(i - 1)
                        continue;
                    end
                    if gini < best_gini
                        best_gini = gini;
                        best_idx = idx;
                        best_thr = (thresholds(i) + thresholds(i - 1)) / 2;
                    end
                end
            end
            idx = best_idx;thr= best_thr;
        end
        if idx ~=-1
            indices_left = X(:, idx) < thr;
            X_left= X(indices_left,:); y_left = y(indices_left,:);
            X_right=X(~indices_left,:); y_right = y(~indices_left,:);
            node.feature_index = idx;
            node.threshold = thr;
            node.left = grow_tree(X_left, y_left, max_depth,n_features_,n_classes_, depth + 1);
            node.right = grow_tree(X_right, y_right, max_depth,n_features_,n_classes_, depth + 1);
        end
    end
end