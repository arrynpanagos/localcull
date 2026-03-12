--[[localcull grouper plugin for darktable

  Groups images by localcull's visual clusters and sets the
  best pick (Red label) as group leader.

  USAGE:
    1. Run localcull on your shoot first (produces *_features.csv)
    2. Import the shoot into darktable
    3. Open lighttable, click "localcull: group by cluster"
    4. Images are now grouped by visual similarity cluster
       - Group leaders = Red picks (consensus best)
       - Expand any group to see alternatives
       - Purple = disagreement alternative pick
       - Green = technical gate failure

  The plugin reads the features CSV to find cluster assignments,
  matches images by filename, and uses darktable's native grouping.
]]

local dt = require "darktable"
local du = require "lib/dtutils"
local df = require "lib/dtutils.file"

du.check_min_api_version("7.0.0", "localcull_grouper")

local MOD = "localcull_grouper"

local gettext = dt.gettext.gettext
local function _(msgid)
  return gettext(msgid)
end

-- return data structure for script_manager
local script_data = {}
script_data.metadata = {
  name = _("localcull grouper"),
  purpose = _("group images by localcull visual clusters"),
  author = "localcull",
  help = "https://github.com/your-repo/localcull"
}
script_data.destroy = nil
script_data.destroy_method = nil
script_data.restart = nil
script_data.show = nil

local DC = {}
DC.module_installed = false
DC.event_registered = false

local GUI = {
  csv_path = {},
  group_btn = {},
  ungroup_btn = {},
  status = {},
}

-- ── CSV parsing ──

local function parse_csv_line(line)
  local fields = {}
  for field in line:gmatch("([^,]*)") do
    table.insert(fields, field)
  end
  return fields
end

local function read_features_csv(csv_path)
  -- Returns table: filename -> {cluster_id, is_selected, is_disagreement, is_tech_fail, star_rating}
  local f = io.open(csv_path, "r")
  if not f then
    return nil, "Cannot open CSV: " .. csv_path
  end

  -- Parse header to find column indices
  local header_line = f:read("*line")
  if not header_line then
    f:close()
    return nil, "Empty CSV file"
  end

  local headers = parse_csv_line(header_line)
  local col = {}
  for i, name in ipairs(headers) do
    col[name] = i
  end

  -- Verify required columns exist
  local required = {"filename", "cluster_id", "star_rating",
                     "dpp_selected", "technical_gate_pass", "aesthetic_disagreement"}
  for _, name in ipairs(required) do
    if not col[name] then
      f:close()
      return nil, "Missing required column: " .. name
    end
  end

  local images = {}
  local n = 0
  for line in f:lines() do
    local fields = parse_csv_line(line)
    if #fields >= #headers then
      local filename = fields[col["filename"]]
      local cluster_id = tonumber(fields[col["cluster_id"]]) or -1
      local star = tonumber(fields[col["star_rating"]]) or 0
      local selected = fields[col["dpp_selected"]] == "True"
      local tech_fail = fields[col["technical_gate_pass"]] == "False"
      local disagreement = fields[col["aesthetic_disagreement"]] == "True"

      images[filename] = {
        cluster_id = cluster_id,
        star_rating = star,
        is_selected = selected,
        is_tech_fail = tech_fail,
        is_disagreement = disagreement,
      }
      n = n + 1
    end
  end
  f:close()

  return images, n .. " images loaded from CSV"
end

-- ── Color label helpers ──
-- darktable color labels: red, yellow, green, blue, purple

local function set_color_label(image, info)
  -- Clear all color labels first
  image.red = false
  image.yellow = false
  image.green = false
  image.blue = false
  image.purple = false

  -- Apply localcull's label (priority: Red > Green > Purple)
  if info.is_selected and not info.is_disagreement then
    image.red = true
  elseif info.is_selected and info.is_disagreement then
    image.purple = true
  elseif info.is_tech_fail then
    image.green = true
  end

  -- Set star rating
  if info.star_rating >= 1 and info.star_rating <= 5 then
    image.rating = info.star_rating
  end
end

-- ── Main grouping logic ──

local function find_csv_for_collection()
  -- Try to auto-detect the features CSV from the current collection
  if #dt.collection == 0 then return nil end

  local first_image = dt.collection[1]
  local dir = first_image.path

  -- Look for *_features.csv in the image directory
  local handle = io.popen('ls "' .. dir .. '"/*_features.csv 2>/dev/null')
  if handle then
    local result = handle:read("*line")
    handle:close()
    if result and result ~= "" then
      return result
    end
  end

  return nil
end

local function do_grouping()
  -- Get CSV path
  local csv_path = GUI.csv_path.text
  if csv_path == nil or csv_path == "" then
    -- Try auto-detect
    csv_path = find_csv_for_collection()
    if csv_path then
      GUI.csv_path.text = csv_path
    else
      dt.print("No CSV path specified and auto-detect failed")
      return
    end
  end

  -- Read CSV
  local csv_data, msg = read_features_csv(csv_path)
  if not csv_data then
    dt.print("Error: " .. msg)
    return
  end
  dt.print_log("localcull: " .. msg)

  -- Build filename -> darktable image lookup from collection
  local dt_images = {}
  for i, image in ipairs(dt.collection) do
    dt_images[image.filename] = image
  end

  -- Group by cluster_id (visual similarity clusters from stage4)
  -- First pass: organize images by cluster and find leaders
  local clusters = {}  -- cluster_id -> list of {dt_image, info}
  local matched = 0
  local unmatched = 0

  for filename, info in pairs(csv_data) do
    local dt_image = dt_images[filename]
    if dt_image then
      local cid = info.cluster_id
      if cid >= 0 then  -- -1 means no cluster assigned
        if not clusters[cid] then
          clusters[cid] = {}
        end
        table.insert(clusters[cid], {image = dt_image, info = info})
      end
      matched = matched + 1
    else
      unmatched = unmatched + 1
    end
  end

  -- Second pass: for each cluster, group images and set leader
  local n_groups = 0
  local n_leaders = 0

  for cluster_id, members in pairs(clusters) do
    if #members < 2 then
      -- Single image, just set labels
      if #members == 1 then
        set_color_label(members[1].image, members[1].info)
      end
      goto continue
    end

    -- Find the best leader: selected + not disagreement (Red pick)
    -- Fallback: highest star rating
    local leader = nil
    local leader_star = -1

    for _, m in ipairs(members) do
      set_color_label(m.image, m.info)

      if m.info.is_selected and not m.info.is_disagreement then
        -- Primary Red pick — best leader candidate
        if not leader or m.info.star_rating > leader_star then
          leader = m
          leader_star = m.info.star_rating
        end
      end
    end

    -- Fallback: if no Red pick found, use highest-rated selected image
    if not leader then
      for _, m in ipairs(members) do
        if m.info.is_selected then
          if not leader or m.info.star_rating > leader_star then
            leader = m
            leader_star = m.info.star_rating
          end
        end
      end
    end

    -- Last fallback: highest-rated image in scene
    if not leader then
      for _, m in ipairs(members) do
        if m.info.star_rating > leader_star then
          leader = m
          leader_star = m.info.star_rating
        end
      end
    end

    if leader then
      -- Make leader the group leader first
      leader.image:make_group_leader()
      n_leaders = n_leaders + 1

      -- Group all other images with the leader
      for _, m in ipairs(members) do
        if m.image ~= leader.image then
          m.image:group_with(leader.image)
        end
      end
      n_groups = n_groups + 1
    end

    ::continue::
  end

  local status_msg = string.format(
    "localcull: %d groups created, %d leaders set, %d matched, %d unmatched",
    n_groups, n_leaders, matched, unmatched
  )
  dt.print(status_msg)
  dt.print_log(status_msg)
  GUI.status.label = status_msg
end

local function do_ungrouping()
  -- Remove all groups from current collection
  local count = 0
  for i, image in ipairs(dt.collection) do
    image:make_group_leader()  -- each image becomes its own leader = ungrouped
    count = count + 1
  end
  local msg = string.format("localcull: ungrouped %d images", count)
  dt.print(msg)
  GUI.status.label = msg
end

-- ── GUI setup ──

GUI.csv_path = dt.new_widget("entry"){
  text = "",
  placeholder = _("path to *_features.csv (auto-detected if empty)"),
  tooltip = _("path to the localcull features CSV file"),
}

GUI.group_btn = dt.new_widget("button"){
  label = _("localcull: group by cluster"),
  tooltip = _("group images by localcull visual clusters, Red pick = group leader"),
  clicked_callback = function() do_grouping() end,
}

GUI.ungroup_btn = dt.new_widget("button"){
  label = _("localcull: ungroup all"),
  tooltip = _("remove all localcull grouping from current collection"),
  clicked_callback = function() do_ungrouping() end,
}

GUI.status = dt.new_widget("label"){
  label = _("ready"),
}

-- ── Module registration ──

local function install_module()
  if not DC.module_installed then
    dt.register_lib(
      MOD,
      _("localcull clusters"),
      true,   -- expandable
      true,   -- resetable
      {[dt.gui.views.lighttable] = {"DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 99}},
      dt.new_widget("box"){
        orientation = "vertical",
        GUI.csv_path,
        GUI.group_btn,
        GUI.ungroup_btn,
        GUI.status,
      }
    )
    DC.module_installed = true
    dt.print_log("localcull grouper module installed")
  end
end

local function destroy()
  dt.gui.libs[MOD].visible = false
end

local function restart()
  dt.gui.libs[MOD].visible = true
end

-- ── Initialization ──

-- Auto-group on film roll import: detect CSV and apply grouping
dt.register_event(
  "localcull_auto_group", "post-import-film",
  function(event, film)
    -- Look for *_features.csv in the imported film's directory
    local dir = film.path
    local handle = io.popen('ls "' .. dir .. '"/*_features.csv 2>/dev/null')
    if not handle then return end
    local csv_path = handle:read("*line")
    handle:close()

    if not csv_path or csv_path == "" then
      dt.print_log("localcull: no features CSV found in " .. dir .. ", skipping auto-group")
      return
    end

    dt.print_log("localcull: found " .. csv_path .. ", auto-grouping...")
    GUI.csv_path.text = csv_path

    -- Small delay to let darktable finish populating dt.collection
    -- then run grouping
    dt.control.sleep(500)
    do_grouping()
  end
)

-- Track last-seen CSV modification time to detect pipeline re-runs
DC.last_csv_mtime = 0
DC.last_csv_path = ""

-- Check for updated CSV when collection changes (e.g. user switches film roll,
-- or re-reads XMP sidecars after a pipeline run)
dt.register_event(
  "localcull_collection_check", "collection-changed",
  function(event)
    if #dt.collection == 0 then return end

    -- Find CSV in current collection's directory
    local dir = dt.collection[1].path
    local handle = io.popen('ls "' .. dir .. '"/*_features.csv 2>/dev/null')
    if not handle then return end
    local csv_path = handle:read("*line")
    handle:close()

    if not csv_path or csv_path == "" then return end

    -- Check if CSV has been modified since last grouping
    local mtime_handle = io.popen('stat -f %m "' .. csv_path .. '" 2>/dev/null')
    if not mtime_handle then return end
    local mtime_str = mtime_handle:read("*line")
    mtime_handle:close()

    local mtime = tonumber(mtime_str) or 0
    if csv_path == DC.last_csv_path and mtime <= DC.last_csv_mtime then
      return  -- CSV unchanged since last grouping
    end

    -- New or updated CSV detected
    DC.last_csv_path = csv_path
    DC.last_csv_mtime = mtime
    GUI.csv_path.text = csv_path

    dt.print("localcull: updated CSV detected, re-grouping...")
    do_grouping()
  end
)

if dt.gui.current_view().id == "lighttable" then
  install_module()
else
  if not DC.event_registered then
    dt.register_event(
      "localcull_grouper", "view-changed",
      function(event, old_view, new_view)
        if new_view.name == "lighttable" and old_view.name == "darkroom" then
          install_module()
        end
      end
    )
    DC.event_registered = true
  end
end

script_data.destroy = destroy
script_data.destroy_method = "hide"
script_data.restart = restart
script_data.show = restart

return script_data