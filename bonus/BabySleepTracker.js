// Variables used by Scriptable.
// These must be at the very top of the file. Do not edit.
// always-run-in-app: true; icon-color: blue;
// icon-glyph: procedures;
var widget = await createWidget();

Script.setWidget(widget);
Script.complete();

async function createWidget(items) {

  let req = new Request("https://192.168.1.123/api/states")
  req.headers = { "Authorization": "Bearer LONG_LIVED_ACCESS_TOKEN", "content-type": "application/json" };
  let json = await req.loadJSON();
  var status = "N/A";
  var start_time = "N/A";
  var duration = "N/A";

  for (var i = 0; i < json.length; i++) {
    if (json[i]['entity_id'] == "baby.sleeping") {
      status = json[i]['state'];
    }

    if (json[i]['entity_id'] == "baby.sleep_time") {

      console.log(json[i])
      start_time = json[i]['state'];

      if (start_time != "N/A") {
        let dateObj = new Date((Math.floor(Date.now() / 1000) - json[i]['attributes']['last_sleeping_time']) * 1000);
        let hours = dateObj.getUTCHours();
        let minutes = dateObj.getUTCMinutes();
        let seconds = dateObj.getSeconds();

        duration = hours.toString().padStart(2, '0')
            + ':' + minutes.toString().padStart(2, '0')
            + ':' + seconds.toString().padStart(2, '0');
      }
    }
  }

  /* Create the widget */
  const widget = new ListWidget();

  if (status == "Sleeping") {  
    const bgColor = new LinearGradient()
    bgColor.colors = [new Color("#077db3"), new Color("#46c1fa")]
    bgColor.locations = [0.0, 1.0]
    widget.backgroundGradient = bgColor

  } else {
    widget.backgroundColor = new Color("#c7c7c7", 1.0);
  }

  const bodyStack = widget.addStack();


  const labelStack = bodyStack.addStack();
  labelStack.setPadding(0, 0, 0, 0);
  labelStack.borderWidth = 0;
  labelStack.layoutVertically();
  labelStack.centerAlignContent()
  labelStack.setPadding(0, 0, 0, 0)

  const my_status = labelStack.addText(status);
  my_status.font = Font.semiboldSystemFont(25);
  my_status.textColor = Color.white();
  my_status.centerAlignText();

  const sleeping_time = labelStack.addText("🛌 " + start_time);
  sleeping_time.font = Font.semiboldSystemFont(22);
  sleeping_time.textColor = Color.white();

  const duration_label = labelStack.addText("⏳ " + duration);
  duration_label.font = Font.semiboldSystemFont(22);
  duration_label.textColor = Color.white();

  return widget;
}
