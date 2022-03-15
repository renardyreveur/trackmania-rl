/*
 * author: Renardy
 * v1.0
 */

namespace Data
{
    // Control print and data collecting condition
    bool playing = true;


    // Class that holds data structure for RL Reward calculation
    class FrameData
    {
        int checkpoint;
        float front_speed;
        float distance;
        int duration;
        bool race_finished;

        string EncodeJSON()
        {
            return "{ \"checkpoint\":" + checkpoint + "," +
                    "\"front_speed\":" + front_speed + "," +
                    "\"distance\":" + distance + "," +
                    "\"duration\":" + duration + "," +
                    "\"race_finished\":" + race_finished + 
                    "}";
        }

    }

    // Method that will run every frame collecting agent state
    string Collect()
    {
        // Get Game context
        auto app = GetApp();
        CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
        
        // Using the Vehicle State Plugin Dependency
        auto visState = VehicleState::ViewingPlayerState();

        // Game artifacts
        auto gameTerminal = playground.GameTerminals[0];
        auto arena = playground.Arena;
        auto player = cast<CSmPlayer>(arena.Players[0]).ScriptAPI;

        // Check if we are playing the game (not in menu etc.)
        if (playground is null || visState is null ||
            !(gameTerminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Playing || 
              gameTerminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish)){
            return "";
        }

        // Get map landmarks (for checkpoint testing)
        MwFastBuffer<CGameScriptMapLandmark@> landmarks = arena.MapLandmarks;

        // Get current in-game time (for race record and duration)
        auto now = cast<CSmArenaRulesMode>(app.PlaygroundScript).Now; // To the closest 10 milliseconds


        // Collect Data while playing
        FrameData fd;
        if (gameTerminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Playing){
            playing = true;
            fd.checkpoint = arena.Players[0].CurrentLaunchedRespawnLandmarkIndex;
            fd.front_speed = visState.FrontSpeed * 3.6f;
            fd.distance = player.Distance;
            fd.duration = now - player.StartTime;
            fd.race_finished = false;
        }

        // Race Finished
        if (gameTerminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish && playing){
            playing = false;
            fd.race_finished = true;
            fd.duration = now - player.StartTime;
            print("Finished the race with a time of " + (now - player.StartTime)/1000.f + " seconds!");
            return fd.EncodeJSON();
        }

        if (playing) {
            return fd.EncodeJSON();
        }else{
            return "";
        }

    }

}