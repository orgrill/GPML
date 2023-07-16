function [RectGrid, params] = CurveGrid2Rect(pxg, pyg)
    % Takes a set of gridded coordinates that are not rectilinear and finds the
    % rectilinear coordinates that are as close as possible in a least squares
    % sense
    numx = size(pxg,1);
    numy = size(pyg,1);
    bigY = [reshape(pyg',numel(pyg),[]); reshape(pyg',numel(pyg),[])];
    bigX = [reshape(pxg',numel(pxg),[])];
    Ag = zeros(length(bigY),numx+1+numy+1);
    Ag(1:numel(pxg),1) = bigX;
    Ag(1:numel(pxg),2:numx+1) = kron(eye(numx),ones(numy,1));
    Ag(numel(pxg)+1:end,numx+2) = bigX;
    Ag(numel(pxg)+1:end,numx+3:end) = repmat(eye(numy),numx,1);
    %params = Ag\bigY;
    params = pinv(Ag)*bigY;
    % Double Check they are orthogonal
    % Tanθ = m1−m2/(1+m1*m2).
%     m1 = params(1);
%     m2 = params(numx+2);
%     theta = atand((m1-m2)/(1+m1*m2));
    % Force Orthogonal
    params(numx+2) = -1/params(1);
    % We actually want the intersections of all the lines we just found
    % take the x coordinates and the shifted x coordinates and evaluate our
    % line equations at each value, then feed this into the intersection
    % function
    BuildevalA = @(x) [x*ones(numx,1) eye(numx) zeros(numx,numy+1); zeros(numy,numx+1) x*ones(numy,1) eye(numy)];
    BigA = arrayfun(BuildevalA,bigX(1),'UniformOutput',false);
    BigA = cell2mat(BigA);
    Ex1 = repelem(bigX(1),numx+numy)';
    Ey1 = BigA*params;
    BigA = arrayfun(BuildevalA,bigX(1)+1,'UniformOutput',false);
    BigA = cell2mat(BigA);
    Ex2 = repelem(bigX(1)+1,numx+numy)';
    Ey2 = BigA*params;
    %AllPerms = num2cell(nchoosek(1:numx+numy,2),2)
    AllPerms = num2cell([repelem(1:numx,numy)' repmat(numx+1:numx+numy,1,numx)'],2);
    
    
    % X = lineXline([Ex1(1:2) Ey1(1:2)],[Ex2(1:2) Ey2(1:2)])
    % scatter(X(1),X(2),'k','filled')
    IntOut = @(x) lineXline([Ex1([x(1) x(2)]) Ey1([x(1) x(2)])],[Ex2([x(1) x(2)]) Ey2([x(1) x(2)])]);
    AllInts = cellfun(IntOut,AllPerms,'UniformOutput',false);
    RectGrid = cell2mat(AllInts);
    % scatter(AllInts(:,1),AllInts(:,2),'r','filled')
    % hold on
    % scatter(SNAPx,SNAPy,'b')
    % scatter(Ex1(1:2),Ey1(1:2),'b','filled')
    % scatter(Ex2(1:2),Ey2(1:2),'r','filled')

end

function [X,P,R,x,p,l] = lineXline(A,B)

    %Find intersection of N lines in D-dimensional space, in least squares sense.
    % X = lineXline(A,B)     -line starts & ends as N*D
    % X = lineXline({x y..})    -lines as D long cell of starts & ends as 2*N
    % [X,P,R] = lineXline(..)      -extra outputs
    % [X,P,R,x,t,l] = lineXline(..)   -plot outputs
    %X: Intersection point, in least squares sense, as 1*D
    %P: Nearest point to the intersection on each line, as N*D
    %R: Distance from intersection to each line, as N*1
    %x: Intersection point X as D-long cell of coordinates {x y..}
    %p: Nearest points P as D-long cell of coordinates {x y..} as N*1
    %l: Initial lines A-to-B as D-long cell of coordinates {x y..} as 2*N
    %
    %Remarks:
    %-Lines are assumed to be infinite in both directions.
    %-Finds point nearest to all lines using minimum sum of squared distances.
    %-For parallel lines returns an arbitrary point and a warning.
    %-For lines of length zero returns NaNs.
    %-Comments/issues/corrections email Serge: s3rg3y@hotmail.com
    %
    %Example: (3 lines, 2 dimensions)
    % [X,P,R,x,p,l] = lineXline([0 0;3 0;0 4],[5 5;0 5;5 2]);
    % plot(x{:},'*k',p{:},'.k',l{:})
    % X2 = lineXline(l) %alternate input form, same results
    %
    %Example: (2 lines, 3 dimensions)
    % [X,P,R,x,p,l] = lineXline(rand(2,3),rand(2,3));
    % plot3(x{:},'*k',p{:},'.k',l{:})
    %
    %Example: (4 lines, 5 dimensions)
    % [X,P,R] = lineXline(rand(4,5),rand(4,5))
    %
    %See also: mldivide
     
    %convert cell input {x y..} to A,B form
    if iscell(A)
        A = permute(cat(3,A{:}),[2 3 1]); %2*N*D > N*D*2
        [A,B] = deal(A(:,:,1),A(:,:,2));
    end
     
    %find intersection
    V = B - A; %vectors from A to B
    V = bsxfun(@rdivide,V,sqrt(sum(V.*V,2))); %normalized vectors
    [N,D] = size(A); %number of points & dimensions
    T = bsxfun(@minus,bsxfun(@times,V',reshape(V,[1 N D])),reshape(eye(D),[D 1 D])); %V.*V-1 as D*N*D
    S = reshape(sum(T,2),[D D]); %sum T along N, as D*D
    C = reshape(T,[D N*D])*A(:); %T*A, as D*1
    X = mldivide(S,C)'; %solve for X: S*X=C, in least squares sense
     
    %checks
    if any(isnan(V(:))) %zero length lines
        warning('lineXline:ZeroLengthLine','One or more lines with zero length.')
    elseif rcond(S)<eps*1000 %parallel lines, stackoverflow.com/questions/13145948/how-to-find-out-if-a-matrix-is-singular
        warning('lineXline:ParallelLines','Lines are near parallel.')
    end
     
    %extra outputs
    if nargout>=2
        U = sum(bsxfun(@times,bsxfun(@minus,X,A),V),2); %dot(X-A,V) distance from A to nearest point on each line
        P = A + bsxfun(@times,U,V); %nearest point on each line
    end
    if nargout>=3
        R = sqrt(sum(bsxfun(@minus,X,P).^2,2)); %distance from intersection to each line
    end
     
    %plot outputs
    if nargout>=4
        x = num2cell(X); %intersection point X
    end
    if nargout>=5
        p = num2cell(P,1); %tangent points P
    end
    if nargout>=6
        l = mat2cell([A(:) B(:)]',2,ones(1,D)*N); %initial lines A,B using cell format {x y..}
    end
end