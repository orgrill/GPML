function result = GSML_CleanWeatherStation(WS_Data)
    % Clean the weather station data, remove NANs
    % Create a fit curve using the low frequency information (hopefully yearly and slower) 
    % Fix any data mismatch where we have weather data but no GCM data

    % First Remove any NAN values
    GoodWeatherData = ~isnan([WS_Data{6}]);
    for i=1:length(WS_Data)
        WS_Data{i} = WS_Data{i}(GoodWeatherData);
    end
    % Some of the Weather Station Seems to be in K, this technique will
    % miss these                
    [StationID, ia, ib] = unique([WS_Data{2}]);
    StationID = string(StationID);
    result = cell(1,6);
    PerStation = cell(length(ia),2);
    for i=1:length(StationID)
        theseMeas = find(ib==i);
        PerStation{i} = {WS_Data{1}(theseMeas), WS_Data{6}(theseMeas)};
        [~, idx] = sort(PerStation{i}{1,1});
        OrigIndex = theseMeas(idx);
        TheseDays = daysact(PerStation{i}{1}(idx(1))-1, PerStation{i}{1}(idx));
        DesiredF = 1:TheseDays(end);
        X_fast = (nufft(PerStation{i}{2}(idx),TheseDays, (0:(length(DesiredF)-1))/length(DesiredF)));
%             figure
%             plot(TheseDays,PerStation{i}{2}(idx))
        N = length(X_fast);
            m = 1:N;
            myHz = m./(N*1);
%             figure
%             plot(myHz,abs((X_fast)),'.-') 
        % Find the largest non-dc component, should be yearlyish, then take all frequencies slower than that 
        [~, MainFreq] = maxk(abs(X_fast(2:end-1)),2);
        FreqMap = zeros(length(X_fast),1);
        FreqMap([transpose(1:min(MainFreq)); transpose(max(MainFreq):length(X_fast))]) = 1;
        X2 = X_fast;
        X2(~FreqMap) = 0;
        % hold on
        % plot(myHz,abs(fftshift(X2)),'.-')                
        Seasonal = real(ifft(X2));%,TheseDays(end));
        TempMismatch = abs(Seasonal(TheseDays)-PerStation{i}{2}(idx));
        %ZTemp = (TempMismatch-mean(TempMismatch))./std(TempMismatch);
        GoodTemp = TempMismatch<30; % This is a pretty loose bound degrees F,
%             figure
%             plot(1:length(Seasonal),Seasonal,TheseDays,PerStation{i}{2}(idx))
%             figure
%             plot(TheseDays,ZTemp)
        % Now Rebuild the WS_Data object
        result(i,:) = {WS_Data{1}(OrigIndex(GoodTemp)) ,WS_Data{2}(OrigIndex(GoodTemp)), WS_Data{3}(OrigIndex(GoodTemp)),WS_Data{4}(OrigIndex(GoodTemp)),WS_Data{5}(OrigIndex(GoodTemp)), WS_Data{6}(OrigIndex(GoodTemp))}; 
    end
    result = [{cat(1,result{:,1})},{cat(1,result{:,2})},{cat(1,result{:,3})},{cat(1,result{:,4})},{cat(1,result{:,5})},{cat(1,result{:,6})}];
end