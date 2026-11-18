"""Contextual help from the installed central guide; no model/factory execution."""
from .common import CLIError, Outcome, command


def register(sub):
    parser=command(sub,'docs','Read or explicitly open installed offline documentation.',run,export=False)
    actions=parser.add_subparsers(dest='action',required=True)
    for action,help in [('list','List stable topic IDs and titles.'),
                        ('show','Show one localized topic and code examples as inert text.'),
                        ('open','Explicitly open one local guide topic in the browser.')]:
        p=command(actions,action,help,run,export=False)
        if action!='list':p.add_argument('topic',help='Exact topic ID; use hyso docs list to discover IDs.')
        p.add_argument('--lang','--language',dest='lang',choices=('tr','en','de','es'),default='tr',help='Guide language (default: tr).')
        if action=='open':p.add_argument('--dry-run',action='store_true',help='Print the topic URI without opening a browser.')


def run(args):
    from hyso_guide import list_topics,show_topic,guide_uri
    try:
        if args.action=='list':
            rows=list_topics(args.lang)
            return Outcome(rows,text=''.join(t['id']+'\t'+t['title']+'\n' for t in rows))
        if args.action=='show':
            topic=show_topic(args.topic,args.lang)
            return Outcome(topic,text=topic['text']+'\n\n'+topic['uri']+'\n')
        uri=guide_uri(args.topic,args.lang)
    except (OSError,ValueError,KeyError) as error:
        raise CLIError(str(error),suggestion='Use hyso docs list --lang '+args.lang+' to find a valid installed topic.') from error
    if args.dry_run:
        return Outcome(dict(uri=uri,opened=False),status='dry_run',text=uri+'\n')
    import webbrowser
    try:
        opened=webbrowser.open(uri)
    except (OSError,webbrowser.Error) as error:
        raise CLIError('The local guide could not be opened: '+str(error),suggestion='Open this local URI manually: '+uri) from error
    if not opened:
        raise CLIError('The browser did not accept the local topic URI.',suggestion='Open this local URI manually: '+uri)
    return Outcome(dict(uri=uri,opened=True),text=uri+'\n')
